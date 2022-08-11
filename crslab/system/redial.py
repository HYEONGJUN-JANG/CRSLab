# @Time   : 2020/12/4
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

# UPDATE
# @Time   : 2021/1/3
# @Author : Xiaolei Wang
# @email  : wxl1999@foxmail.com

import torch
from loguru import logger

from crslab.data import dataset_language_map
from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class ReDialSystem(BaseSystem):
    """This is the system for KGSF model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(ReDialSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                           restore_system, interact, debug, tensorboard)
        self.ind2tok = vocab['conv']['ind2tok']
        self.end_token_idx = vocab['conv']['end']
        self.item_ids = side_data['rec']['item_entity_ids']
        self.id2entity = vocab['rec']['id2entity']

        self.rec_optim_opt = opt['rec']
        self.conv_optim_opt = opt['conv']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

        self.language = dataset_language_map[self.opt['dataset']]

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        assert stage in ('rec', 'conv')
        assert mode in ('train', 'valid', 'test')

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        if stage == 'rec':
            rec_loss, rec_scores = self.rec_model.forward(batch, mode=mode)
            rec_loss = rec_loss.sum()
            if mode == 'train':
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_scores, batch['item'])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        else: # Conversation
            # gen_loss, preds , batch_response = self.conv_model.forward(batch, mode=mode)
            gen_loss, preds = self.conv_model.forward(batch, mode=mode) # TODO : DEFAULT


            # HJ
            # batch_response = batch['response'].view(-1)
            # [self.ind2tok.get(i) for i in batch['request'][0][:batch['request_lengths'][0]].tolist()]  # request
            # batch response [self.ind2tok.get(i) for i in batch_response.tolist()[:15]]


            gen_loss = gen_loss.sum()
            if mode == 'train':
                self.backward(gen_loss)
            else:
                self.conv_evaluate(preds, batch['response'])
            gen_loss = gen_loss.item()
            self.evaluator.optim_metrics.add('gen_loss', AverageMetric(gen_loss))
            self.evaluator.gen_metrics.add('ppl', PPLMetric(gen_loss))
            return preds

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.rec_model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader['rec'].get_rec_data(batch_size=self.rec_batch_size):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')  # report train loss
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['rec'].get_rec_data(batch_size=self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')  # report valid loss
                # early stop
                metric = self.evaluator.optim_metrics['rec_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['rec'].get_rec_data(batch_size=self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        self.init_optim(self.conv_optim_opt, self.conv_model.parameters())
        templist=[]
        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size,
                                                                         shuffle=False):
                    self.step(batch, stage='conv', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')
                metric = self.evaluator.optim_metrics['gen_loss']
                # if epoch%2==0:
                #     templist.append(self.printConv(batch,preds))
                if self.early_stop(metric):
                    break

        # test
        logger.info('[Test-Redial]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                preds = self.step(batch, stage='conv', mode='test')
                templist.append(self.printConv(batch, preds)) # HJ : For Conv 저장용
            self.evaluator.report(mode='test')
        self.saveConv(templist) # HJ : For Conv 저장용

    def fit(self):
        self.train_recommender()
        self.train_conversation()



    def interact(self):
        pass

    def print2dTokArr(self, idxlist, tdic=None, padtok=0):
        '''
        :param idxlist: 2-dim Tensor
        :param tdic: key-value dictionary, key is int
        :param padtok: padding token idx which if skipped
        :return: text list composed with idx-list-token value without padding
        '''
        # if idxlist.size()
        tlist=[]
        if tdic:
            for i in idxlist:
                txt=''
                for t in list(filter(lambda x: x!=padtok, i)):
                    txt+= f'{tdic.get(t)} '
                if txt: tlist.append(txt.rstrip())
        else:
            for i in idxlist:
                txt=''
                for t in list(filter(lambda x:x!=padtok, i)): ## 0 일경우 '__pad__' 니까 넘겨버리자
                        txt+=f'{self.ind2tok.get(t)} '
                if txt: tlist.append(txt.rstrip())
        return tlist

    def printConv(self,batch,preds):
        '''
        :param batch: batch (dict) 를 그대로 받는것
        :param preds: preds로 step의 결과인 token index list (2-dim)
        :return: 3-dim array [B,[context(1d list),response(text),pred(text)]]
        '''
        uttrlist = []
        # 원본문장?
        batchlen=batch['context'].shape[0]
        for i in range(batchlen):
            texts = self.print2dTokArr(batch['context'][i].tolist())
            req=self.list2Text([self.ind2tok.get(i) for i in list(filter(lambda x : x, batch.get('request')[i].tolist()))])
            repn=self.list2Text([self.ind2tok.get(i) for i in list(filter(lambda x : x, batch.get('response')[i].tolist()))])
            pred=self.list2Text([self.ind2tok.get(i) for i in preds[i].tolist()])
            if texts[-1] == req: # Redial의 경우 context의 마지막발화가 request이고, 그것에 대한 응답이 response
                uttrlist.append([texts,repn,pred])
        return uttrlist

    def list2Text(self,wlist):
        txt=''
        for i in wlist:
            if i: txt+=f'{i} '
            else: txt+=f'<TOK> '
        return txt.rstrip()

    def saveConv(self,convlist):
        try:
            path=f"./convlog_{str(self.model.module).split('(')[0]}.txt"
        except: # For Redial Error (No Module)
            path=f"./convlog_{str(self).split()[0].split('.')[-1]}.txt"
        with open(path,'w') as f:
            for i in convlist:
                for k in i:
                    context, response, pred = k
                    f.write("<< Context >>\n")
                    for cont in context:
                        f.write(f"{cont}\n")
                    f.write(f"\n<<Real Response>> : {response}\n")
                    f.write(f"<<Created Response>>: {pred}\n")
                    f.write("\n================< NEW LINE >================\n\n")
        print(f"Conversation Saved in {path}")


