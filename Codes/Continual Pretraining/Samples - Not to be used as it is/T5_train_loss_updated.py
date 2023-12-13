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

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

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
    def calculate_masked_digit_score(predictions, targets, masks, penalty_factor=0.9):
        total_score = 0
        num_sentences = len(predictions)

        for i in range(num_sentences):
            prediction = predictions[i]
            target = targets[i]
            mask = masks[i]

            masked_digit_count = sum(mask)
            match_count = 0
            penalty_count = 0

            for j in range(len(mask)):
                if mask[j] == 1:
                    if j < len(prediction) and j < len(target):
                        if prediction[j] == target[j]:
                            match_count += 1
                        else:
                            penalty_count += 1
                    elif j >= len(prediction):
                        penalty_count += 1

            sentence_score = match_count / masked_digit_count if masked_digit_count > 0 else 1.0

            # Apply penalty factor for non-digit predictions in masked values
            sentence_score *= penalty_factor ** penalty_count

            total_score += sentence_score

        average_score = total_score / num_sentences if num_sentences > 0 else 0.0
        return average_score

    def calculate_length_difference(predictions, targets):
        """
        Calculate the difference between the length of predicted and actual output sentences.

        Args:
            predictions (List[str]): List of predicted output sentences.
            targets (List[str]): List of target output sentences.

        Returns:
            float: Average difference between the lengths of predicted and actual output sentences.
        """
        total_difference = 0
        num_sentences = len(predictions)

        for i in range(num_sentences):
            prediction = predictions[i]
            target = targets[i]

            prediction_length = len(prediction)
            target_length = len(target)

            difference = target_length - prediction_length
            if difference > 0:
                total_difference += difference

        average_difference = total_difference / num_sentences if num_sentences > 0 else 0.0
        return average_difference


    @staticmethod
    def calculate_scores(predictions, ground_truths):
        em_score = 0
        subset_match_score = 0

        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  Metric.exact_match_score(prediction, ground_truth)
            subset_match_score += Metric.approx_match_score(prediction, ground_truth)
        length_difference = Metric.calculate_length_difference(predictions, ground_truths)
        em_score /= len(predictions)
        subset_match_score /= len(predictions)
        return em_score*100, subset_match_score*100, length_difference


class T5Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path, max_length=512)
        self.tokenizer.add_tokens("<mask>")
        self.em_score_list = []
        self.subset_score_list = []
        

        # self.freeze_params(self.model.model.encoder, excluded_layers=['layer0', 'layer1']) ## freezing the layer0 and 1 of encoder
        # self.freeze_embeds()

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

    def _step(self, batch):
        lm_labels = batch["target_ids_actual"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        logits = outputs["logits"]

        # Calculate the cross-entropy loss
        ce_loss = outputs[0]

        # Compute the length difference penalty
        pred_lengths = (logits.argmax(dim=-1) != self.tokenizer.pad_token_id).sum(dim=1)
        actual_lengths = (batch["target_ids_actual"] != self.tokenizer.pad_token_id).sum(dim=1)
        length_diff = pred_lengths - actual_lengths
        length_penalty = torch.square(length_diff.float()).mean()  # Penalize larger differences more

        # Combine the cross-entropy loss and length difference penalty
        alpha = 30.0  # Adjust the weighting between the two terms
        loss = ce_loss * alpha * length_penalty
        
        '''
        # Compute the masked digit score penalty
        preds = self.ids_to_clean_text(outputs["logits"])
        targets = self.ids_to_clean_text(batch["target_ids_actual"])
        masks = batch["mask"]
        masked_digit_score = Metric.calculate_masked_digit_score(preds, targets, masks, penalty_factor=2)
        masked_digit_penalty = torch.tensor(masked_digit_score)

        # Add the masked digit score penalty to the loss
        loss += masked_digit_penalty        
        '''
        
        return loss

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

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
        targets = self.ids_to_clean_text(batch["target_ids_actual"])

        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]

        loss = self._step(batch)
        base_metrics = {'val_loss': loss}
        summ_len = np.mean(self.lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=targets)

        mask = batch["mask"]
        masked_preds = [pred if mask[i].cpu().numpy().all() == 0 else "<mask>" for i, pred in enumerate(preds)]

        masked_digit_score = Metric.calculate_masked_digit_score(masked_preds, targets, mask)
        em_score, subset_match_score, length_difference = Metric.calculate_scores(preds, targets)

        em_score = torch.tensor(em_score, dtype=torch.float32)
        subset_match_score = torch.tensor(subset_match_score, dtype=torch.float32)

        base_metrics.update(em_score=em_score, subset_match_score=subset_match_score)
        self.log("masked_digit_score", masked_digit_score, sync_dist=True)
        self.log("length_difference", length_difference, sync_dist=True)
        self.log("val_loss", loss, sync_dist=True)
        self.log("em_score", em_score, sync_dist=True)
        self.log("subset_match_score", subset_match_score, sync_dist=True)
        return base_metrics

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        self.log("train_loss", loss, on_step=False, on_epoch=True)
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
        self.tokenizer.add_tokens("<mask>")
        self.input_length = input_length
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return len(self.dataset)  # Return the total number of samples in the dataset

    def clean_text(self, text):
        text = text.replace("\n", "")
        text = text.replace("``", "")
        text = text.replace('"', "")
        text = text.lower()
        return text

    def mask_numerical_digits(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        masked_tokens = []
        mask = [0] * len(tokens)

        # Find the indices of the digits
        digit_indices = [i for i, token in enumerate(tokens) if re.match(r"\d+", token)]

        if len(digit_indices) < 2:
            return sentence, mask

        first_digit_index = digit_indices[0]
        second_digit_index = digit_indices[1]
        last_digit_index = digit_indices[-1]

        # Mask 50% of the digits between the first two digits and the last digit
        digits_to_mask = digit_indices[2:-1]
        num_digits_to_mask = int(0.7 * len(digits_to_mask))
        digits_to_mask = random.sample(digits_to_mask, num_digits_to_mask)

        for i, token in enumerate(tokens):
            if i in digits_to_mask:
                masked_tokens.append("<mask>")
                mask[i] = 1
            else:
                masked_tokens.append(token)

        # Add special tokens to the mask
        tokenized_sentence = self.tokenizer.convert_tokens_to_string(masked_tokens)
        final_tokens = self.tokenizer.tokenize(tokenized_sentence)
        mask += [0] * (len(final_tokens) - len(mask))

        masked_sentence = " ".join(final_tokens)

        # Pad the mask to match the desired length
        mask = mask[:self.output_length]
        mask += [0] * (self.output_length - len(mask))

        return masked_sentence, mask

    def convert_to_features(self, example_batch):
        question = self.clean_text(example_batch["question"])
        answer = self.clean_text(example_batch["answer"])

        # Mask numerical digits in the answer
        masked_answer, mask = self.mask_numerical_digits(answer)

        input_encoding = self.tokenizer.encode_plus(
            question,
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        output_encoding_masked = self.tokenizer.encode_plus(
            masked_answer,
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        output_encoding_actual = self.tokenizer.encode_plus(
            answer,
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        mask = mask[:self.output_length]
        mask += [0] * (self.output_length - len(mask))

        return input_encoding, output_encoding_masked, output_encoding_actual, mask

    def __getitem__(self, index):
        input_encoding, output_encoding_masked, output_encoding_actual, mask = self.convert_to_features(
            self.dataset.iloc[index]
        )

        source_ids = input_encoding["input_ids"].squeeze()
        source_mask = input_encoding["attention_mask"].squeeze()
        target_ids_masked = output_encoding_masked["input_ids"].squeeze()
        target_ids_actual = output_encoding_actual["input_ids"].squeeze()
        target_mask = output_encoding_masked["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids_masked": target_ids_masked,
            "target_ids_actual": target_ids_actual,
            "target_mask": target_mask,
            "mask": mask,
        }


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
            train_df = pd.read_csv(self.train_file)
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
    parser.add_argument('--output_length', default=512)
    parser.add_argument('--num_train_epochs', default=30)
    parser.add_argument('--output_dir', default='t5_pretraining')
    parser.add_argument('--train_batch_size', default=4)
    parser.add_argument('--learning_rate', default=1e-5)
    parser.add_argument('--model', default='google/t5-v1_1-large')
    hparam = parser.parse_args()

    args_dict = dict(
        output_dir="", # path to save the checkpoints
        model_name_or_path=hparam.model,
        tokenizer_name_or_path=hparam.model,
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
    log_dir = os.path.join('logs', 'T5_loss_update_base')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # create a logger instance
    logger = TensorBoardLogger(save_dir='logs/', name='T5_loss_update_large')
       
    # Save the model and configuration
    checkpoint_dir = 'checkpoints/T5_loss_update_large/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='T5_loss_update-best',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_weights_only=False,
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

    data_module = TextDataModule(train_file = 'gsm_train.csv', val_file = 'gsm_val.csv', test_file = 'gsm_val.csv', tokenizer_name_or_path = args.tokenizer_name_or_path, input_length= args.max_input_length, output_length= args.max_output_length, batch_size=4, num_workers=8)
    #set_seed(42)
    model = T5Model(args)
    

    trainer = pl.Trainer(**train_params)
    trainer.fit(model, data_module)
    
    # Save the configuration separately
    #model.config.save_pretrained(checkpoint_dir + '/config')   
    #trainer.callbacks.append(checkpoint_callback)
    model.model.save_pretrained(checkpoint_dir)
    model.tokenizer.save_pretrained(checkpoint_dir)
