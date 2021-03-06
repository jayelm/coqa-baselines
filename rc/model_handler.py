import time
from utils.data_utils import prepare_datasets
from utils import constants as Constants
from model import Model
import torch
import os
import json
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.logger import ModelLogger
from utils.eval_utils import AverageMeter
from utils.data_utils import sanitize_input, vectorize_input, sanitize_input_dialog_batched, vectorize_input_dialog_batched


class ModelHandler(object):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """

    def __init__(self, config):
        self.logger = ModelLogger(config, dirname=config['dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname
        cuda = config['cuda']
        cuda_id = config['cuda_id']
        if not cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if cuda_id < 0 else 'cuda:%d' % cuda_id)

        datasets = prepare_datasets(config)
        train_set = datasets['train']
        dev_set = datasets['dev']
        test_set = datasets['test']

        # Evaluation Metrics:
        self._train_loss = AverageMeter()
        self._train_f1 = AverageMeter()
        self._train_em = AverageMeter()
        self._dev_f1 = AverageMeter()
        self._dev_em = AverageMeter()

        if train_set:
            self.train_loader = DataLoader(train_set, batch_size=config['batch_size'],
                                           shuffle=config['shuffle'], collate_fn=lambda x: x, pin_memory=True)
            self._n_train_batches = len(train_set) // config['batch_size']
        else:
            self.train_loader = None

        if dev_set:
            self.dev_loader = DataLoader(dev_set, batch_size=config['batch_size'],
                                         shuffle=False, collate_fn=lambda x: x, pin_memory=True)
            self._n_dev_batches = len(dev_set) // config['batch_size']
        else:
            self.dev_loader = None

        if test_set:
            self.test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False,
                                          collate_fn=lambda x: x, pin_memory=True)
            self._n_test_batches = len(test_set) // config['batch_size']
            self._n_test_examples = len(test_set)
        else:
            self.test_loader = None

        self._n_train_examples = 0
        self.model = Model(config, train_set)
        self.model.network = self.model.network.to(self.device)
        self.config = self.model.config
        self.is_test = False

    def train(self):
        if self.train_loader is None or self.dev_loader is None:
            print("No training set or dev set specified -- skipped training.")
            return

        self.is_test = False
        timer = Timer("Train")
        self._epoch = self._best_epoch = 0

        if self.dev_loader is not None:
            print("\n>>> Dev Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self._run_epoch(self.dev_loader, training=False, verbose=self.config['verbose'])
            timer.interval("Validation Epoch {}".format(self._epoch))
            format_str = "Validation Epoch {} -- F1: {:0.2f}, EM: {:0.2f} --"
            print(format_str.format(self._epoch, self._dev_f1.mean(), self._dev_em.mean()))

        self._best_f1 = self._dev_f1.mean()
        self._best_em = self._dev_em.mean()
        if self.config['save_params']:
            self.model.save(self.dirname)
        self._reset_metrics()

        while self._stop_condition(self._epoch):
            self._epoch += 1

            print("\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self._run_epoch(self.train_loader, training=True, verbose=self.config['verbose'])
            train_epoch_time = timer.interval("Training Epoch {}".format(self._epoch))
            format_str = "Training Epoch {} -- Loss: {:0.4f}, F1: {:0.2f}, EM: {:0.2f} --"
            print(format_str.format(self._epoch, self._train_loss.mean(),
                  self._train_f1.mean(), self._train_em.mean()))

            print("\n>>> Dev Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self._run_epoch(self.dev_loader, training=False, verbose=self.config['verbose'])
            timer.interval("Validation Epoch {}".format(self._epoch))
            format_str = "Validation Epoch {} -- F1: {:0.2f}, EM: {:0.2f} --"
            print(format_str.format(self._epoch, self._dev_f1.mean(), self._dev_em.mean()))

            if self._best_f1 <= self._dev_f1.mean():  # Can be one of loss, f1, or em.
                self._best_epoch = self._epoch
                self._best_f1 = self._dev_f1.mean()
                self._best_em = self._dev_em.mean()
                if self.config['save_params']:
                    self.model.save(self.dirname)
                print("!!! Updated: F1: {:0.2f}, EM: {:0.2f}".format(self._best_f1, self._best_em))

            self._reset_metrics()
            self.logger.log(self._train_loss.last, Constants._TRAIN_LOSS_EPOCH_LOG)
            self.logger.log(self._train_f1.last, Constants._TRAIN_F1_EPOCH_LOG)
            self.logger.log(self._train_em.last, Constants._TRAIN_EM_EPOCH_LOG)
            self.logger.log(self._dev_f1.last, Constants._DEV_F1_EPOCH_LOG)
            self.logger.log(self._dev_em.last, Constants._DEV_EM_EPOCH_LOG)
            self.logger.log(train_epoch_time, Constants._TRAIN_EPOCH_TIME_LOG)

        timer.finish()
        self.training_time = timer.total

        print("Finished Training: {}".format(self.dirname))
        print(self.summary())

    def test(self):
        if self.test_loader is None:
            print("No testing set specified -- skipped testing.")
            return

        self.is_test = True
        self._reset_metrics()
        timer = Timer("Test")
        output = self._run_epoch(self.test_loader, training=False, verbose=0,
                                 out_predictions=self.config['out_predictions'],
                                 out_attentions=self.config['save_attn_weights'])

        if self.config['dialog_batched']:
            # Slightly different id format
            _id = None
            turn = 0
            for ex in output:
                if ex['id'] != _id:
                    _id = ex['id']
                    turn = 0
                ex['id'] = _id
                ex['turn_id'] = turn
                turn += 1
        else:
            for ex in output:
                _id = ex['id']
                ex['id'] = _id[0]
                ex['turn_id'] = _id[1]

        if self.config['out_predictions']:
            output_file = os.path.join(self.dirname, Constants._PREDICTION_FILE)
            with open(output_file, 'w') as outfile:
                json.dump(output, outfile, indent=4)
                if self.config['out_predictions_csv']:
                    import pandas as pd
                    for o in output:
                        o['gold_answer_1'], o['gold_answer_2'], o['gold_answer_3'], o['gold_answer_4'] = o['gold_answers']
                    output_csv = pd.DataFrame(output)
                    output_csv = output_csv[[
                        'id', 'turn_id', 'span_start', 'span_end',
                        'answer',
                        'gold_answer_1',
                        'gold_answer_2',
                        'gold_answer_3',
                        'gold_answer_4',
                        'f1', 'em'
                    ]]
                    output_csv.to_csv(output_file.replace('.json', '.csv'),
                                      index=False)

        test_f1 = self._dev_f1.mean()
        test_em = self._dev_em.mean()

        timer.finish()
        print(self.report(self._n_test_batches, None, test_f1, test_em, mode='test'))
        self.logger.log([test_f1, test_em], Constants._TEST_EVAL_LOG)
        print("Finished Testing: {}".format(self.dirname))

    def _run_epoch(self, data_loader, training=True, verbose=10, out_predictions=False, out_attentions=None):
        start_time = time.time()
        output = []
        for step, input_batch in enumerate(data_loader):
            if self.config['dialog_batched']:
                x_batches = []
                for ib in input_batch:
                    ib_sanitized = sanitize_input_dialog_batched(ib, self.config, self.model.word_dict,
                                                                 self.model.feature_dict, training=training)
                    x_batch = vectorize_input_dialog_batched(ib_sanitized, self.config, training=training,
                                                             device=self.device)
                    if not x_batch:
                        continue  # When there are no target spans present in the batch
                    x_batches.append(x_batch)
            else:
                input_batch = sanitize_input(input_batch, self.config, self.model.word_dict,
                                        self.model.feature_dict, training=training)
                x_batch = vectorize_input(input_batch, self.config, training=training,
                                     device=self.device)
                if not x_batch:
                    continue  # When there are no target spans present in the batch
                x_batches = [x_batch]  # Singleton list.

            res = self.model.predict(x_batches, update=training, out_predictions=out_predictions, out_attentions=out_attentions)

            loss = res['loss']
            f1 = res['f1']
            em = res['em']
            total_ex = sum(xb['batch_size'] for xb in x_batches)
            self._update_metrics(loss, f1, em, total_ex, training=training)

            if training:
                self._n_train_examples += total_ex

            if (verbose > 0) and (step % verbose == 0):
                mode = "train" if training else ("test" if self.is_test else "dev")
                print(self.report(step, loss, f1 * 100, em * 100, mode))
                print('used_time: {:0.2f}s'.format(time.time() - start_time))

            if out_predictions:
                for id, prediction, span, f1, em, ans in zip(res['ids'], res['predictions'], res['spans'], res['f1s'], res['ems'], res['answers']):
                    output.append({'id': id,
                                   'answer': prediction,
                                   'span_start': span[0],
                                   'span_end': span[1],
                                   'f1': f1,
                                   'em': em,
                                   'gold_answers': ans})
        return output

    def report(self, step, loss, f1, em, mode='train'):
        if mode == "train":
            format_str = "[train-{}] step: [{} / {}] | exs = {} | loss = {:0.4f} | f1 = {:0.2f} | em = {:0.2f}"
            return format_str.format(self._epoch, step, self._n_train_batches, self._n_train_examples, loss, f1, em)
        elif mode == "dev":
            return "[predict-{}] step: [{} / {}] | f1 = {:0.2f} | em = {:0.2f}".format(
                    self._epoch, step, self._n_dev_batches, f1, em)
        elif mode == "test":
            return "[test] | test_exs = {} | step: [{} / {}] | f1 = {:0.2f} | em = {:0.2f}".format(
                    self._n_test_examples, step, self._n_test_batches, f1, em)
        else:
            raise ValueError('mode = {} not supported.' % mode)

    def summary(self):
        start = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = "Best epoch = {}\nDev F1 = {:0.2f}\nDev EM = {:0.2f}".format(
            self._best_epoch, self._best_f1, self._best_em)
        end = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    def _update_metrics(self, loss, f1, em, batch_size, training=True):
        if training:
            self._train_loss.update(loss, batch_size)
            self._train_f1.update(f1 * 100, batch_size)
            self._train_em.update(em * 100, batch_size)
        else:
            self._dev_f1.update(f1 * 100, batch_size)
            self._dev_em.update(em * 100, batch_size)

    def _reset_metrics(self):
        self._train_loss.reset()
        self._train_f1.reset()
        self._train_em.reset()
        self._dev_f1.reset()
        self._dev_em.reset()

    def _stop_condition(self, epoch):
        """
        Checks have not exceeded max epochs and has not gone 10 epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + 10
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True
