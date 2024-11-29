# Prepare data for conformal prediction. Train the base model using availiable data up until now.
import torch.nn as nn
import torch
from epiweeks import Week
import yaml
import numpy as np
import random
import argparse
from tqdm import tqdm
from datetime import timedelta

from utils import ForecasterTrainer, EarlyStopping, pickle_save, pickle_load, decode_onehot, last_nonzero
from seq2seq import Seq2seq
from transformer import TransformerEncoderDecoder
from load_covid import prepare_data, prepare_region_fine_tuning_data
from load_power import prepare_power_data, convert_to_datetime, datetime2str, load_all_power_data, prepare_testing_power_data
from metrics import rmse, norm_rmse, mape

import warnings
warnings.filterwarnings("ignore")


def test(model, model_name, test_dataloader, device, true_scale, ys_scalers):
    model.eval()
    predictions = {}
    addition_info = {}
    with torch.no_grad():
        for batch in test_dataloader:
            # get data
            regions, meta, x, x_mask, y, y_mask, weekid = batch
            x_mask = x_mask.type(torch.float)
            regionid = decode_onehot(meta)

            if model_name == 'seq2seq':
                # send to device
                meta = meta.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)

                # forward pass
                y_pred, emb = model.forward(x, x_mask, meta, output_emb=True)
                emb = emb.cpu().numpy()
            
            if model_name == 'transformer':
                weekid = last_nonzero(weekid)
                # send to device
                regionid = regionid.to(device)
                weekid = weekid.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)

                # forward pass
                y_pred = model.forward(x, x_mask, regionid, weekid).unsqueeze(-1)
                # empty emb
                emb = np.zeros(len(regions))

            # convert to numpy
            y_pred = y_pred.cpu().numpy()
            y = y.numpy()
            y = y[:, :, 0]
            meta = meta.cpu().numpy()

            # use scaler to inverse transform
            for i, region in enumerate(regions):
                if region not in predictions:
                    predictions[region] = []
                    addition_info[region] = []
                if true_scale:
                    predictions[region].append(ys_scalers[region].inverse_transform(y_pred[i]).reshape(-1))
                    y_in_true_scale = ys_scalers[region].inverse_transform(y[i].reshape(-1, 1)).reshape(-1)
                    addition_info[region].append((y_in_true_scale, y_mask[i], x[i], x_mask[i], weekid[i]))
                else:
                    predictions[region].append(y_pred[i].reshape(-1))
                    addition_info[region].append((y[i], y_mask[i], x[i], x_mask[i], weekid[i]))
            print(len(predictions['X']))
    return predictions, addition_info


def train_and_test(last_train_time, params, pretrained_model_state, train=True, power_df=None):
    device = torch.device(params['device'])
    true_scale = params['true_scale']
    
    if params['dataset'] == 'power':
        params['last_train_time'] = datetime2str(last_train_time)
        train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_testing_power_data(power_df, params)
        x_dim += 1
    elif params['dataset'] == 'covid':
        params['last_train_time'] = Week.fromstring(last_train_time).cdcformat()
        train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_data(params)
    elif params['dataset'] == 'weather':
        params['last_train_time'] = datetime2str(last_train_time)
        train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_testing_power_data(power_df, params)
        
    metas_dim = len(params['regions'])

    # create model
    model = None
    if params['model_name'] == 'seq2seq':
        model = Seq2seq(
            metas_train_dim=metas_dim,
            x_train_dim=x_dim-1,
            device=device,
            weeks_ahead=params['weeks_ahead'],
            hidden_dim=params['model_parameters']['hidden_dim'],
            out_layer_dim=params['model_parameters']['out_layer_dim'],
            out_dim=1
        )
    if params['model_name'] == 'transformer':
        model = TransformerEncoderDecoder(
            input_dim=x_dim-1,
            output_dim=params['weeks_ahead'],
            hidden_dim=params['model_parameters']['hidden_dim'],
            seq_length=seq_length,
            num_layers=params['model_parameters']['num_layers'],
            num_heads=params['model_parameters']['num_heads'],
            num_regions=metas_dim,
            rnn_hidden_dim=params['model_parameters']['rnn_hidden_dim'],
            rnn_layers=params['model_parameters']['rnn_layers'],
        )
    model = model.to(device)
    
    if train or pretrained_model_state is None:
        epochs = params['training_parameters']['epochs']
        if params['week_retrain'] == False and pretrained_model_state is not None:
            model.load_state_dict(pretrained_model_state)
            epochs = params['week_retrain_epochs']

        # create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=params['training_parameters']['lr'])

        # create loss function
        loss_fn = nn.MSELoss()

        # create scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=params['training_parameters']['gamma'],
            verbose=False)

        # create early stopping
        early_stopping = EarlyStopping(
            patience=100, verbose=False)

        # create trainer
        trainer = ForecasterTrainer(model, params['model_name'], optimizer, loss_fn, device)

        # train model
        for epoch in range(epochs):
            trainer.train(train_dataloader, epoch)
            val_loss = trainer.evaluate(val_dataloader, epoch)
            scheduler.step()
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        
        pretrained_model_state = early_stopping.model_state_dict

    model.load_state_dict(pretrained_model_state)
    
    predictions = {}
    addition_info = {}
    
    predictions, addition_info = test(model, params['model_name'], test_dataloader, device, true_scale, ys_scalers)
    return predictions, addition_info, pretrained_model_state


def get_params(input_file='1'):
    with open(f'../../setup/exp_params/{input_file}.yaml', 'r') as stream:
        try:
            ot_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    
    data_params_file = '../../setup/covid_mortality.yaml'
    if ot_params['dataset']:
        if ot_params['dataset'] == 'power':
            data_params_file = '../../setup/power.yaml'
        elif ot_params['dataset'] == 'weather':
            data_params_file = '../../setup/weather.yaml'
    
    with open(data_params_file, 'r') as stream:
        try:
            task_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    with open('../../setup/seq2seq.yaml', 'r') as stream:
        try:
            model_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    # merge params
    params = {**task_params, **model_params}
    
    # overwrite using online training params
    for key, value in ot_params.items():
        params[key] = value

    if params['dataset'] == 'covid':
        params['data_params']['start_week'] = Week.fromstring(params['data_params']['start_week']).cdcformat()
        params['test_week'] = Week.fromstring(str(params['test_week'])).cdcformat()
    
    print('Paramaters loading success.')
    
    return params


def run(params):
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    
    base_pred = []
    test_pred = None
    pretrained_model_state = None
    
    if params['dataset'] == 'power':
        starting_time = str(params['pred_starting_time'])
        test_time = str(params['test_time'])
        test_time = convert_to_datetime(test_time)
        power_df = load_all_power_data()
        # get base model predictions
        current_time = convert_to_datetime(starting_time)
        predictions, addition_infos, pretrained_model_state = train_and_test(current_time, params, pretrained_model_state, power_df=power_df)
    else:
        starting_week = str(params['pred_starting_week'])
        current_week = Week.fromstring(starting_week).cdcformat()
        predictions, addition_infos, pretrained_model_state = train_and_test(current_week, params, pretrained_model_state)
    base_pred.append((predictions, addition_infos))
    
    results = {
        'params': params,
        'base_pred': base_pred,
    }
    return results


def eval_results(data_id, steps_ahead=4):
    results = pickle_load(f'../../results/base_pred/saved_pred_{data_id}.pickle', version5=True)
    predictions, addition_infos = results['base_pred'][0]
    regions = list(predictions.keys())
    formatted_pred = {}
    for region in regions:
        formatted_pred[region] = {}
        for step in range(steps_ahead):
            formatted_pred[region][step] = {
                'true': [],
                'pred': [],
            }
    for region in regions:
        prediction = predictions[region]
        addition_info = addition_infos[region]
        for i in range(len(prediction)):
            y_preds = prediction[i]
            y_trues = addition_info[i][0]
            for w in range(steps_ahead):
                formatted_pred[region][w]['true'].append(y_trues[w])
                formatted_pred[region][w]['pred'].append(y_preds[w])
    for region in regions:
        for step in range(steps_ahead):
            y_preds = np.array(formatted_pred[region][step]['pred'])
            y_trues = np.array(formatted_pred[region][step]['true'])
            rmse_val = rmse(y_preds, y_trues)
            nrmse_val = norm_rmse(y_preds, y_trues)
            mape_val = mape(y_preds, y_trues)
            print(f'{region}, {step}: rmse: {rmse_val}, nrmse: {nrmse_val}, mape: {mape_val}')
            


if __name__ == '__main__':
    # only run one step in online training. The train, validation and test datasets are randomly split.
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Input file")
    parser.add_argument('--eval', '-e', action='store_true')
    args = parser.parse_args()
    input_file = args.input    # for example: 1
    
    params = get_params(input_file)
    data_id = int(params['data_id'])
    
    if args.eval:
        eval_results(data_id)
    else:
        results = run(params)
        pickle_save(f'../../results/base_pred/saved_pred_{data_id}.pickle', results)