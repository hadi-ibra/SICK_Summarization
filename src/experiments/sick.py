from typing import Any

import numpy as np
import nltk
import torch
from tqdm import tqdm
from overrides import overrides
from datasets import load_metric
from numpy.random._generator import Generator as Generator
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import DialogsumDataset
from src.experiments.experiment import BasicExperiment
from src.logging.logger import Logger


class SickExperiment(BasicExperiment):

    def __init__(
        self,
        model,
        finetune_args: Seq2SeqTrainingArguments,
        freeze_encoder: bool,
        train_ds: Dataset,
        eval_ds: Dataset,
        test_ds: Dataset,
        tokenizer,
        gen: Generator,
        device,
        logger: Logger,
    ) -> None:
        super().__init__(model, train_ds, eval_ds, test_ds, tokenizer, gen, device, logger)
        self.finetune_args = finetune_args
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self._freeze_weight(model.get_encoder())

        self.finetune_trainer = Seq2SeqTrainer(
            model=model,
            args=finetune_args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics,
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        self.is_test_ds_dialog_sum = False
        self.metric = load_metric("src/utils/rouge.py")
        self.bertscore_metric = load_metric("bertscore", lang="en", model_type="bert-base-uncased")
        if isinstance(test_ds, DialogsumDataset):
            self.is_test_ds_dialog_sum = True
            self.metric2 = load_metric("src/utils/rouge.py")
            self.metric3 = load_metric("src/utils/rouge.py")
            self.bertscore_metric = load_metric("bertscore", lang="en", model_type="bert-base-uncased")
            self.bertscore_metric = load_metric("bertscore", lang="en", model_type="bert-base-uncased")

    @overrides
    def save(self) -> Any:
        return self.finetune_trainer

    @overrides
    def train(self) -> None:
        self.finetune_trainer.train()

    @overrides
    def test(self, **kwargs) -> None:
        self.model = self.model.to(self.device)
        self.model.eval()
        test_dataloader = DataLoader(dataset=self.test_ds, batch_size=1, shuffle=False)

        total_decoded_preds = []
        total_decoded_labels = []

        with torch.no_grad():
            for idx, data in enumerate(tqdm(test_dataloader), 0):
                x = data["input_ids"].to(self.device, dtype=torch.long)
                mask = data["attention_mask"].to(self.device, dtype=torch.long)
                y = data["labels"].to(self.device, dtype=torch.long)

                num_beams = kwargs.get("num_beams", 20)
                generated_ids = self.model.generate(
                    input_ids=x, attention_mask=mask, max_length=100, num_beams=num_beams
                )

                generated_ids = generated_ids.cpu()
                y = y.cpu()

                decoded_preds = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                y = np.where(y != -100, y, self.tokenizer.pad_token_id)
                decoded_labels = self.tokenizer.batch_decode(
                    y, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                # Rouge expects a newline after each sentence
                decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
                decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
                self.metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                self.bertscore_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

                if self.is_test_ds_dialog_sum:
                    y2 = data["labels2"].to(self.device, dtype=torch.long)
                    y2 = y2.cpu()
                    y3 = data["labels3"].to(self.device, dtype=torch.long)
                    y3 = y3.cpu()

                    decoded_labels2 = self.tokenizer.batch_decode(
                        y2, skip_special_tokens=True, clean_up_tokenization_space=True
                    )
                    decoded_labels3 = self.tokenizer.batch_decode(
                        y3, skip_special_tokens=True, clean_up_tokenization_space=True
                    )

                    decoded_labels2 = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels2]
                    decoded_labels3 = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels3]

                    # add_batch should not return anything, investigate why they wrote it in the original paper code
                    result2 = self.metric2.add_batch(predictions=decoded_preds, references=decoded_labels2)
                    result3 = self.metric3.add_batch(predictions=decoded_preds, references=decoded_labels3)

                    self.bertscore_metric2.add_batch(predictions=decoded_preds, references=decoded_labels2)
                    self.bertscore_metric3.add_batch(predictions=decoded_preds, references=decoded_labels3)

                total_decoded_preds.append(decoded_preds)
                total_decoded_labels.append(decoded_labels)

        bertscore_result = self.bertscore_metric.compute(lang="en", model_type="bert-base-uncased")
        result = self.metric.compute(use_stemmer=True)

        if self.is_test_ds_dialog_sum:
            result2 = self.metric2.compute(use_stemmer=True)
            result3 = self.metric3.compute(use_stemmer=True)

            bertscore_result2 = self.bertscore_metric2.compute(lang="en", model_type="bert-base-uncased")
            bertscore_result3 = self.bertscore_metric3.compute(lang="en", model_type="bert-base-uncased")

        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        bertscore_result = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])

        if self.is_test_ds_dialog_sum:
            bertscore_result2 = sum(bertscore_result2["f1"]) / len(bertscore_result2["f1"])
            bertscore_result3 = sum(bertscore_result3["f1"]) / len(bertscore_result3["f1"])

            result2 = {key: value.mid.fmeasure * 100 for key, value in result2.items()}
            result3 = {key: value.mid.fmeasure * 100 for key, value in result3.items()}

        self.logger.save_results(result)
        self.logger.save_results({"bert_score": bertscore_result})
        # self.logger.save_results({"decoded_pred": total_decoded_preds})
        # self.logger.save_results({"decoded_label": total_decoded_labels})

    def _freeze_weight(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
