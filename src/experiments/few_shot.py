import sys
from typing import List

import numpy as np
import copy

from overrides import overrides
from rouge import Rouge
from bert_score import score
from tqdm import tqdm

from src.experiments.experiment import BasicExperiment


class FewShotLearning(BasicExperiment):

    def __init__(
        self,
        model,
        train_ds,
        eval_ds,
        test_ds,
        tokenizer,
        temperature,
        k,
        gen: np.random.Generator,
        device,
        logger,
        is_test_ds_dialog_sum,
    ) -> None:
        super().__init__(model, train_ds, eval_ds, test_ds, tokenizer, gen, device, logger)
        self.temperature = temperature
        self.k = k
        self.dialog_max_lenght = 2048
        self.use_temperature = True if temperature > 0 else False
        self.model.resize_token_embeddings(len(tokenizer))
        self.is_test_ds_dialog_sum = is_test_ds_dialog_sum
        if self.k != 0:
            examples = self._get_training_samples()
            self.base_prompt = self._gen_examples_template(examples)
        else:
            self.base_prompt = "Summarize the following chat dialog in one sentence.\nDIALOG:"

        tokens = self.tokenizer.tokenize(str(self.base_prompt))
        token_count = len(tokens)
        self.length_max = token_count + self.dialog_max_lenght

    @overrides
    def train(self) -> None:
        pass
        # raise NotImplementedError(
        #     "Training for an LLM is not implemented since the required hardware capabilieties are too high"
        # )

    @overrides
    def test(self, **kwargs) -> None:
        summaries = []

        for dialog, summary_gold in tqdm(self.test_ds):
            prompt = self.base_prompt + dialog + "\nSUMMARY:"
            inputs = self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(
                **inputs,
                do_sample=self.use_temperature,
                temperature=self.temperature,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=self.length_max,
            )
            output = self.tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            output_summary = output.split("SUMMARY:")[-1].strip().strip("\n")
            if not output_summary:
                output_summary = "model empty generation"
            summaries.append((output_summary, summary_gold))

        self.logger.save_results({"summaries": summaries})
        metrics = self._compute_metrics(summaries)
        self.logger.save_results(metrics)
        self.logger.summary(metrics)

    @overrides
    def save(self):
        pass

    def _get_training_samples(self):
        idxes = self.gen.integers(0, len(self.train_ds), size=self.k)
        examples = []
        for idx in idxes:
            dialog, summary = self.train_ds[idx]
            examples.append((dialog, summary))
        return examples

    def _gen_examples_template(self, training_examples: List[str]) -> str:
        header = "Summarize the chat dialog. Here you can find some examples:\n"
        tail = "Summarize the following chat dialog in one sentence.\nDIALOG:"
        examples = []
        for dialog, summary in training_examples:
            template_example = f"DIALOG: {dialog}SUMMARY: {summary}\n"
            examples.append(template_example)
        return header + " ".join(examples) + tail

    def _compute_metrics(self, summaries) -> dict:
        sys.setrecursionlimit(self.length_max + 10)
        rouge = Rouge()
        if self.is_test_ds_dialog_sum:
            model_summaries = []
            gold_summaries1 = []
            gold_summaries2 = []
            gold_summaries3 = []
            for s in summaries:
                model_summaries.append(s[0])
                gold_summaries1.append(s[1][0])
                gold_summaries2.append(s[1][1])
                gold_summaries3.append(s[1][2])
            gold_summaries_types = [gold_summaries1, gold_summaries2, gold_summaries3]
        else:
            model_summaries, gold_summaries1 = map(list, zip(*[s for s in summaries]))
            gold_summaries_types = [gold_summaries1]

        result = {}

        for idx, gold_summaries in enumerate(gold_summaries_types):
            score_tot = rouge.get_scores(model_summaries, gold_summaries, avg=True)
            p_b, r_b, f_b = score(
                model_summaries,
                gold_summaries,
                lang="en",
                model_type="microsoft/deberta-large-mnli",
                batch_size=1,
                device=self.device,
            )
            p_bert = p_b.mean()
            r_bert = r_b.mean()
            f_bert = f_b.mean()

            score_tot["bert"] = {
                "r": r_bert.detach().cpu().numpy().tolist(),
                "p": p_bert.detach().cpu().numpy().tolist(),
                "f": f_bert.detach().cpu().numpy().tolist(),
            }

            result[f"summary_{idx}"] = copy.deepcopy(score_tot)

        return result
