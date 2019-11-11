import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from dataset.dataset_generation_testset import data_preprocessing,writefile
import pandas as pd
def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=1,
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2
    # )
    

    # build model architecture
    dataset = data_preprocessing('dataset/')
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    #loss_fn = getattr(module_loss, config['loss'])
    #metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False, do_lower_case =False)
    #total_loss = 0.0
    #total_metrics = torch.zeros(len(metric_fns))
    samples = pd.read_csv('dataset/testset.csv')
    ranking = [0]*len(samples)
    for idx in range(len(samples)):
        scene = samples['shot_id'].values[idx].split('_')[0]
        shot_id = int(samples['shot_id'].values[idx].split('_')[1])
        question = samples['question'].values[idx]
        writefile(dataset,'dataset/test.csv',scene,question)

        data_loader = module_data.FriendsBertDataLoader('dataset/test.csv',1, shuffle=False, validation_split=0.0, num_workers=1, training=False,tokenizer = tokenizer)
        with torch.no_grad():
            L = []
            for batch_idx, data in enumerate(data_loader):

                input_data = [data['input1'][0],data['input2'][0],data['input3'][0],data['input4'][0]]
                input_segment = [data['input1'][1],data['input2'][1],data['input3'][1],data['input4'][1]]
                
                for i in range(4):
                    input_data[i] = input_data[i].to(device)
                    input_segment[i] = input_segment[i].to(device)

                #self.optimizer.zero_grad()
                output = model((input_data[0],input_segment[0]),(input_data[1],input_segment[1]),(input_data[2],input_segment[2]),(input_data[3],input_segment[3]))
                #print(scene,shot_id,question,output[0][0][1])
                L.append(output[0][0][1].item())
            obj = pd.Series(L)
            #print(L,obj.rank(method=max)[shot_id-1],shot_id)
            #print(type())
            ranking[obj.rank(method='max')[shot_id-1].astype(int)]+=1
            #print(ranking)
    for i in range(1,len(ranking)):
        ranking[i] += ranking[i-1]
    for i in range(1,len(ranking)):
        ranking[i]=ranking[i]/len(samples)
    print(ranking)
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)

    #         #
    #         # save sample images, or do something with output here
    #         #

    #         # computing loss, metrics on test set
    #         loss = loss_fn(output, target)
    #         batch_size = data.shape[0]
    #         total_loss += loss.item() * batch_size
    #         for i, metric in enumerate(metric_fns):
    #             total_metrics[i] += metric(output, target) * batch_size

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
