import torch
from torch import nn
import random
import pandas as pd
import numpy as np
import math
import re
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn import functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
from string import punctuation
import string
import argparse
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

''' some score metrics that would be used in our experiments'''
class Metric:
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())
        '''
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        '''
        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(lower(s)))
    
    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return int(Metric.normalize_answer(prediction) == Metric.normalize_answer(ground_truth))

    def approx_match_score(prediction, ground_truth):
        answer = Metric.normalize_answer(prediction) 
        gt = Metric.normalize_answer(ground_truth)
        match = 0
        gt_words = gt.split(" ")
        for word in gt_words:
            if word in answer:
                match = 1
                return match
        return match
    
    @staticmethod
    def calculate_scores(predictions, ground_truths):
        em_score = 0
        subset_match_score = 0

        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  Metric.exact_match_score(prediction, ground_truth)
            subset_match_score += Metric.approx_match_score(prediction, ground_truth)
        em_score /= len(predictions)
        subset_match_score /= len(predictions)
        return em_score*100, subset_match_score*100


class T5Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path, max_length=512)
        self.em_score_list = []
        self.subset_score_list = []
        
    # Freezing some of the layers may be helpful when we are training on a small dataset and it is not very different from the pretrained dataset
    def freeze_params(self, model, excluded_layers=None):
        if excluded_layers is None:
            excluded_layers = []
        for name, param in model.named_parameters():
            if not any(layer_name in name for layer_name in excluded_layers):
                param.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
   
    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        # Calculate the cross-entropy loss
        ce_loss = outputs[0]

        # Calculate the KL divergence
        kl_div_loss = F.kl_div(
            F.log_softmax(outputs.logits, dim=-1),
            F.softmax(outputs.logits, dim=-1),
            reduction='batchmean'
        )
        
        # Adjust the weight factor for the KL divergence loss
        kl_div_weight = 1  # Adjust the weight factor as desired


        # Combine the cross-entropy loss and the KL divergence
        loss = ce_loss + kl_div_weight * kl_div_loss

        return loss, ce_loss, kl_div_loss

    def _generative_step(self, batch):
        t0 = time.time()
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch["target_mask"],
            max_length=512,
            num_beams=3,
            early_stopping=True,
            temperature=0.7
        )
        preds = self.ids_to_clean_text(generated_ids)
        targets = self.ids_to_clean_text(batch["target_ids"])

        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]

        loss, ce_loss, kl_div_loss = self._step(batch)
        base_metrics = {'val_loss': loss}
        base_metrics.update(preds=preds, target=targets)

        em_score, subset_match_score = Metric.calculate_scores(preds, targets)

        em_score = torch.tensor(em_score, dtype=torch.float32)
        subset_match_score = torch.tensor(subset_match_score, dtype=torch.float32)

        base_metrics.update(em_score=em_score, subset_match_score=subset_match_score)

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_ce_loss", ce_loss, sync_dist=True)
        self.log("kl_div_loss", kl_div_loss, sync_dist=True)
        self.log("em_score", em_score, sync_dist=True)
        self.log("subset_match_score", subset_match_score, sync_dist=True)
        return base_metrics

    def training_step(self, batch, batch_idx):
        loss, ce_loss, kl_div_loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        # Log the metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_ce_loss", ce_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train kl_div_loss", kl_div_loss, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss, "log": tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.05)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True),
            "monitor": "val_loss"  # Change 'metric_to_track' to the desired metric for monitoring
        }
        return [optimizer], [lr_scheduler]


class TextData(Dataset):
    def __init__(self, df, tokenizer, input_length, output_length, print_text=False):
        self.dataset = df
        self.tokenizer = tokenizer
        #self.tokenizer.add_tokens("<mask>")
        self.input_length = input_length
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return len(self.dataset)  # Return the total number of samples in the dataset

    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n', ' ')
        text = text.replace('``', '')
        text = text.replace('"', '')
        return text

    def remove_values_less_than(self, numbers, threshold):
        return [num for num in numbers if num >= threshold]

    def select_random_percentage(self, lst, percentage):
        num_values = int(len(lst) * (percentage / 100))
        random_values = random.sample(lst, num_values)
        sorted_values = sorted(random_values)
        return sorted_values

    def mask_numeric_spans(self, text, mask_ratio, noise_span_length):
        tokens = text.split()
        mask_started = False
        masked_tokens = []
        mask = [0] * len(tokens)

        for i, token in enumerate(tokens):
            if token == "[Answer]":
                start_index = i+1

        digit_indices = [j for j, token in enumerate(tokens) if re.match(r'\d+', token)]    
        mask_digit_index = self.remove_values_less_than(digit_indices, start_index)
        mask_digit_index = mask_digit_index[:-2]
        random_mask_index = self.select_random_percentage(mask_digit_index, mask_ratio)

        for i in random_mask_index:
            for j in range(noise_span_length):
                if j < noise_span_length:
                    mask[i+j] = 1
                else:
                    mask[i-1] = 1

        return mask


    def noise_span_to_unique_sentinel(self, text, mask, sentinels):
        tokens = text.split()
        text_ = []
        one_count=0
        sentinel_cnt=0
        for i in range(len(tokens)):
            if mask[i] == 1:
                one_count+=1
                if one_count==1:
                    text_.append(sentinels[sentinel_cnt])
                    sentinel_cnt+=1
                else:
                    if one_count==3:
                        one_count=0
            else:
                text_.append(tokens[i])
        text_ = ' '.join(text_)
        return text_

    def nonnoise_span_to_unique_sentinel(self, text, mask, sentinels):
        tokens = text.split()
        text_ = []
        zero_first=True
        sentinel_cnt=0
        for i in range(len(tokens)):
            if mask[i] == 0:
                if zero_first:
                    text_.append(sentinels[sentinel_cnt])
                    zero_first=False
                    sentinel_cnt+=1
            else:
                zero_first=True
                text_.append(tokens[i])
        text_ = ' '.join(text_)
        return text_
    
    def convert_to_features(self, example_batch):
        text = self.clean_text(example_batch['text'])
        mask = self.mask_numeric_spans(text, mask_ratio = 50, noise_span_length = 3)
        sentinels = [f'<extra_id_{i}>' for i in range(100)]
        input_ = self.noise_span_to_unique_sentinel(text,mask,sentinels)
        target_ = self.nonnoise_span_to_unique_sentinel(text,mask,sentinels)

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
    
        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


class TextDataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, test_file, tokenizer_name_or_path, input_length, output_length, batch_size=4, num_workers=16):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path)
        self.input_length = input_length
        self.output_length = output_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_df = pd.read_parquet(self.train_file)
            val_df = pd.read_csv(self.val_file)
            self.train_data = TextData(train_df, self.tokenizer, self.input_length, self.output_length)
            self.val_data = TextData(val_df, self.tokenizer, self.input_length, self.output_length)
        if stage == 'test' or stage is None:
            test_df = pd.read_csv(self.test_file)
            self.test_data = TextData(test_df, self.tokenizer, self.input_length, self.output_length)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_length', default=512)
    parser.add_argument('--output_length', default=200)
    parser.add_argument('--num_train_epochs', default=30)
    parser.add_argument('--output_dir', default='t5_pretraining')
    parser.add_argument('--train_batch_size', default=4)
    parser.add_argument('--learning_rate', default=1e-5)
    parser.add_argument('--model', default='google/t5-v1_1-large')
    hparam = parser.parse_args()

    args_dict = dict(
        output_dir="", # path to save the checkpoints
        model_name_or_path=hparam.model,
        tokenizer_name_or_path=hparam.model,#"T5_large_custom_tokenizer",
        max_input_length=int(hparam.input_length),
        max_output_length=int(hparam.output_length),
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=1e-5,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=4,
        eval_batch_size=4,
        num_train_epochs=2,
        gradient_accumulation_steps=1,
        n_gpu=1,
        resume_from_checkpoint=None, 
        val_check_interval = 1.0,
        n_val=0,
        val_percent_check= 0,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default otheriwse 1
        seed=101,
    )

    args_dict.update({'output_dir': hparam.output_dir, 'num_train_epochs':int(hparam.num_train_epochs),
                    'train_batch_size': int(hparam.train_batch_size), 'eval_batch_size': int(hparam.train_batch_size), 'learning_rate': float(hparam.learning_rate)})
    args = argparse.Namespace(**args_dict)

    # Create the logger directory if it doesn't exist
    log_dir = os.path.join('logs', 'T5_loss_update_large')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # create a logger instance
    logger = TensorBoardLogger(save_dir='logs/', name='T5_loss_update_large')
       
    # Save the model and configuration
    checkpoint_dir = 'checkpoints/T5_loss_update_large_1/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='T5_loss_update-best',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
    )


    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.num_train_epochs,
        accelerator='auto',
        precision=16 if args.fp_16 else 32,
        gradient_clip_val=args.max_grad_norm,
        val_check_interval=args.val_check_interval,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    data_module = TextDataModule(train_file = 'train_data.parquet', val_file = 'val_data.csv', test_file = 'gsm_val.csv', tokenizer_name_or_path = args.tokenizer_name_or_path, input_length= args.max_input_length, output_length= args.max_output_length, batch_size=4, num_workers=8)
    #set_seed(42)
    model = T5Model(args)
    

    trainer = pl.Trainer(**train_params)
    trainer.fit(model, data_module)
    
    # Save the configuration separately
    #model.config.save_pretrained(checkpoint_dir + '/config')   
    #trainer.callbacks.append(checkpoint_callback)
    model.model.save_pretrained(checkpoint_dir)
    model.tokenizer.save_pretrained(checkpoint_dir)
