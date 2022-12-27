import argparse
import json
import logging
import os
import random
import collections
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import TestDataset
from model import KGReasoning
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
from util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple
from torchmetrics import SpearmanCorrCoef

query_name_dict = {('e', ('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    ('e', ('r', 'r', 'r', 'r', 'r')): '5p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM',
                }
name_answer_dict = {'1p': ['e', ['r',], 'e'],
                    '2p': ['e', ['r', 'e', 'r'], 'e'],
                    '3p': ['e', ['r', 'e', 'r', 'e', 'r'], 'e'],
                    '2i': [['e', ['r',], 'e'], ['e', ['r',], 'e'], 'e'],
                    '3i': [['e', ['r',], 'e'], ['e', ['r',], 'e'], ['e', ['r',], 'e'], 'e'],
                    'ip': [[['e', ['r',], 'e'], ['e', ['r',], 'e'], 'e'], ['r',], 'e'],
                    'pi': [['e', ['r', 'e', 'r'], 'e'], ['e', ['r',], 'e'], 'e'],
                    '2in': [['e', ['r',], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    '3in': [['e', ['r',], 'e'], ['e', ['r',], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    'inp': [[['e', ['r',], 'e'], ['e', ['r', 'n'], 'e'], 'e'], ['r',], 'e'],
                    'pin': [['e', ['r', 'e', 'r'], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    'pni': [['e', ['r', 'e', 'r', 'n'], 'e'], ['e', ['r',], 'e'], 'e'],
                    '2u-DNF': [['e', ['r',], 'e'], ['e', ['r',], 'e'], ['u',], 'e'],
                    'up-DNF': [[['e', ['r',], 'e'], ['e', ['r',], 'e'], ['u',], 'e'], ['r',], 'e'],
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys()) # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']
espace = 9
rspace = 11
mapping = dict()

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--do_cp', action='store_true', help="do cardinality prediction")
    parser.add_argument('--path', action='store_true', help="do interpretation study")

    parser.add_argument('--train', action='store_true', help="do test")
    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('--kbc_path', type=str, default=None, help="kbc model path")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int, help="used to speed up torch.dataloader")
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--fraction', type=int, default=1, help='fraction the entity to save gpu memory usage')
    parser.add_argument('--thrshd', type=float, default=0.001, help='thrshd for neural adjacency matrix')
    parser.add_argument('--neg_scale', type=int, default=1, help='scaling neural adjacency matrix for negation')
    
    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=12345, type=int, help="random seed")
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    return parser.parse_args(args)

def log_metrics(mode, metrics, writer):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s: %f' % (mode, metric, metrics[metric]))
        print('%s %s: %f' % (mode, metric, metrics[metric]))
        writer.write('%s %s: %f\n' % (mode, metric, metrics[metric]))

def read_triples(filenames, nrelation, datapath):
    adj_list = [[] for i in range(nrelation)]
    edges_all = set()
    edges_vt = set()
    for filename in filenames:
        with open(filename) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                adj_list[int(r)].append((int(h), int(t)))
    for filename in ['valid.txt', 'test.txt']:
        with open(os.path.join(datapath, filename)) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                edges_all.add((int(h), int(r), int(t)))
                edges_vt.add((int(h), int(r), int(t)))
    with open(os.path.join(datapath, "train.txt")) as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            edges_all.add((int(h), int(r), int(t)))

    return adj_list, edges_all, edges_vt

def verify_chain(chain, chain_structure, edges_y, edges_p): # (e, r, e, ..., e)
    '''
    verify the validity of the reasoning path (chain)
    '''
    global mapping
    head = chain[0]
    rel = 0
    neg = False
    judge = True
    edge_class = []
    for ele, ans_ele in zip(chain_structure[1:], chain[1:]):
        if ele == 'e':
            if neg:
                edge_judge = ((head, rel, ans_ele) not in edges_y)
                judge = judge & edge_judge
                if edge_judge: # not in train/val/test
                    edge_class.append('y')
                elif (head, rel, ans_ele) in edges_p: # in val/test
                    edge_class.append('p')
                else: # in train
                    edge_class.append('n')
                neg = False
            else:
                edge_judge = ((head, rel, ans_ele) in edges_y)
                if edge_judge:
                    if (head, rel, ans_ele) in edges_p: # in val/test
                        edge_class.append('p')
                    else: # in train
                        edge_class.append('y')
                else: # not in train/val/test
                    edge_class.append('n')
                judge = judge & edge_judge
            head = ans_ele
        elif ele == 'r':
            rel = ans_ele
        elif ele == 'n':
            neg = True
    
    chain_structure = chain_structure[1:-1]
    chain = chain[1:-1]
    out = ''
    neg = False
    edge_class = edge_class[::-1]
    idx = 0
    for ele, ans_ele in zip(chain_structure[::-1], chain[::-1]):
        if ele == 'e':
            out += '{:<9}'.format(str(ans_ele))
            mapping[str(ans_ele)] = id2ent[ans_ele]
        elif ele == 'r':
            if neg:
                out += '{:<11}'.format(edge_class[idx]+'<-r'+str(ans_ele)+'-X')
                neg = False
            else:
                out += '{:<11}'.format(edge_class[idx]+'<-r'+str(ans_ele)+'-')
            mapping['r'+str(ans_ele)] = id2rel[ans_ele]
            idx += 1
        elif ele == 'n':
            neg = True
    return judge, out

def verify(ans_structure, ans, edges_y, edges_p, offset=0):
    '''
    verify the validity of the reasoning path
    '''
    global mapping
    if ans_structure[1][0] == 'r': # [[...], ['r', ...], 'e']
        chain_stucture = ['e']+ans_structure[1]+['e']
        if ans_structure[0] == 'e': # ['e', ['r', ...], 'e']
            chain = [ans[0]]+ans[1]+[ans[2]]
            judge, out = verify_chain(chain, chain_stucture, edges_y, edges_p)
            out = '{:<9}'.format(str(ans[2])) + out + '{:<9}'.format(str(ans[0]))
            mapping[str(ans[2])] = id2ent[ans[2]]
            mapping[str(ans[0])] = id2ent[ans[0]]
            return judge, out
        else:
            chain = [ans[0][-1]]+ans[1]+[ans[2]]
            judge1, out1 = verify_chain(chain, chain_stucture, edges_y, edges_p)
            for ele in ans_structure[1] + [ans_structure[2]]:
                if ele == 'r':
                    offset += 11
                elif ele == 'e':
                    offset += 9
            judge2, out2 = verify(ans_structure[0], ans[0], edges_y, edges_p, offset)
            judge = judge1 & judge2
            out = '{:<9}'.format(str(ans[2])) + out1 + out2
            mapping[str(ans[2])] = id2ent[ans[2]]
            return judge, out
        
    else: # [[...], [...], 'e']
        if ans_structure[-2][0] == 'u':
            union = True
            out = '{:<9}'.format(str(ans[-1])+'(u)')
            ans_structure, ans = ans_structure[:-1], ans[:-1]
        else:
            union = False
            out = '{:<9}'.format(str(ans[-1])+'(i)')
        mapping[str(ans[-1])] = id2ent[ans[-1]]
        judge = not union
        offset += 9
        for ele, ans_ele in zip(ans_structure[:-1], ans[:-1]):
            judge_ele, out_ele = verify(ele, ans_ele, edges_y, edges_p, offset)
            if union:
                judge = judge | judge_ele
            else:
                judge = judge & judge_ele
            out = out + out_ele + '\n' + ' '*offset
        return judge, out

def get_cp_thrshd(model, tp_answers, fn_answers, args, dataloader, query_name_dict, device):
    '''
    get the best threshold for cardinality prediction on valid set
    '''
    probs = defaultdict(list)
    cards = defaultdict(list)
    best_thrshds = dict()
    for queries, queries_unflatten, query_structures in tqdm(dataloader):
        queries = torch.LongTensor(queries).to(device)
        embedding, _, _ = model.embed_query(queries, query_structures[0], 0)
        embedding = embedding.squeeze()
        hard_answer = tp_answers[queries_unflatten[0]]
        easy_answer = fn_answers[queries_unflatten[0]]
        num_hard = len(hard_answer)
        num_easy = len(easy_answer)

        probs[query_structures[0]].append(embedding.to('cpu'))
        cards[query_structures[0]].append(torch.tensor([num_hard+num_easy]))
    for query_structure in probs:
        prob = torch.stack(probs[query_structure])#.to(device)
        card = torch.stack(cards[query_structure]).squeeze().to(torch.float)#.to(device)
        ape = torch.zeros_like(card).to(torch.float).to(device)
        best_thrshd = 0
        best_mape = 10000
        nquery = prob.size(0)
        fraction = 10
        dim = nquery // fraction
        rest = nquery - fraction * dim
        for i in tqdm(range(10)):
            thrshd = i / 10
            for j in range(fraction):
                s = j * dim
                t = (j+1) * dim
                if j == fraction - 1:
                    t += rest
                fractional_prob = prob[s:t, :].to(device)
                fractional_card = card[s:t].to(device)
                pre_card = (fractional_prob >= thrshd).to(torch.float).sum(-1)
                ape[s:t] = torch.abs(fractional_card - pre_card) / fractional_card
            mape = ape.mean()
            if mape < best_mape:
                best_mape = mape
                best_thrshd = thrshd
        best_thrshds[query_structure] = best_thrshd
    print(best_thrshds)
    return best_thrshds

def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, device, writer, edges_y, edges_p, cp_thrshd):
    '''
    Evaluate queries in dataloader
    '''
    global mapping
    mode = "Test"
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)
    logs = defaultdict(list)
    rates = defaultdict(list)
    probs = defaultdict(list)
    cards = defaultdict(list)
    for queries, queries_unflatten, query_structures in tqdm(dataloader):
        queries = torch.LongTensor(queries).to(device)
        embedding, _, exec_query = model.embed_query(queries, query_structures[0], 0)
        embedding = embedding.squeeze()

        order = torch.argsort(embedding, dim=-1, descending=True)
        ranking = torch.argsort(order)
        # eval
        hard_answer = tp_answers[queries_unflatten[0]]
        easy_answer = fn_answers[queries_unflatten[0]]
        num_hard = len(hard_answer)
        num_easy = len(easy_answer)
        cur_ranking = ranking[list(easy_answer) + list(hard_answer)]
        all_path, h1_path, h3_path, h10_path = 0, 0, 0, 0
        num_h1, num_h3, num_h10 = 0, 0, 0
        if args.path:
            for root in list(hard_answer):
                rank = ranking[root]
                rank -= ((cur_ranking < rank).sum()-1)
                ans, _ = model.find_ans(exec_query, query_structures[0], root)
                mapping = dict()
                judge, out = verify(name_answer_dict[query_name_dict[query_structures[0]]], ans, edges_y, edges_p)
                if judge:
                    all_path += 1
                if rank <= 1:
                    num_h1 += 1
                    if judge:
                        h1_path += 1
                if rank <= 3:
                    num_h3 += 1
                    if judge:
                        h3_path += 1
                if rank <= 10:
                    num_h10 += 1
                    if judge:
                        h10_path += 1
                print(judge, rank.item())
                print(out, mapping)
        if args.do_cp:
            probs[query_structures[0]].append(embedding.to('cpu'))
            cards[query_structures[0]].append(torch.tensor([num_hard+num_easy]))
        cur_ranking, indices = torch.sort(cur_ranking)
        masks_hard = indices >= num_easy
        masks_easy = indices < num_easy
        answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(device)
        cur_ranking = cur_ranking - answer_list + 1 # filtered setting
        cur_ranking_hard = cur_ranking[masks_hard] # take indices that belong to the hard answers
        cur_ranking_easy = cur_ranking[masks_easy] # take indices that belong to the easy answers

        mrr_hard = torch.mean(1./cur_ranking_hard).item()
        h1_hard = torch.mean((cur_ranking_hard <= 1).to(torch.float)).item()
        h3_hard = torch.mean((cur_ranking_hard <= 3).to(torch.float)).item()
        h10_hard = torch.mean((cur_ranking_hard <= 10).to(torch.float)).item()
        mrr_easy = torch.mean(1./cur_ranking_easy).item()
        h1_easy = torch.mean((cur_ranking_easy <= 1).to(torch.float)).item()
        h3_easy = torch.mean((cur_ranking_easy <= 3).to(torch.float)).item()
        h10_easy = torch.mean((cur_ranking_easy <= 10).to(torch.float)).item()
        if num_easy == 0:
            mrr_easy, h1_easy, h3_easy, h10_easy = 1, 1, 1, 1

        logs[query_structures[0]].append({
            'MRR_hard': mrr_hard,
            'HITS1_hard': h1_hard,
            'HITS3_hard': h3_hard,
            'HITS10_hard': h10_hard,
            'num_hard_answer': num_hard,
            'MRR_easy': mrr_easy,
            'HITS1_easy': h1_easy,
            'HITS3_easy': h3_easy,
            'HITS10_easy': h10_easy,
            'num_easy_answer': num_easy,
        })
        if args.path:
            if num_hard > 0:
                rates[query_name_dict[query_structures[0]]+" all path interpretability"].append(all_path / num_hard)
            if num_h1 > 0:
                rates[query_name_dict[query_structures[0]]+" HITS1 path interpretability"].append(h1_path / num_h1)
            if num_h3 > 0:
                rates[query_name_dict[query_structures[0]]+" HITS3 path interpretability"].append(h3_path / num_h3)
            if num_h10 > 0:
                rates[query_name_dict[query_structures[0]]+" HITS10 path interpretability"].append(h10_path / num_h10)
    if args.path:
        rate_metric = defaultdict(float)
        for query_structure in rates:
            rate_metric[query_structure] = sum(rates[query_structure])/len(rates[query_structure])
        log_metrics('Interpretability', rate_metric, writer)

    metrics = collections.defaultdict(lambda: collections.defaultdict(int))
    for query_structure in logs:
        for metric in logs[query_structure][0].keys():
            if metric in ['num_hard_answer', 'num_easy_answer']:
                continue
            metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
        metrics[query_structure]['num_queries'] = len(logs[query_structure])
    
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode+" "+query_name_dict[query_structure], metrics[query_structure], writer)
        for metric in metrics[query_structure]:
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average'%mode, average_metrics, writer)

    if args.do_cp:
        card_metrics = defaultdict(float)
        spearman = SpearmanCorrCoef()
        for query_structure in probs:
            prob = torch.stack(probs[query_structure])
            card = torch.stack(cards[query_structure]).squeeze().to(torch.float)
            pre_card = (prob >= cp_thrshd[query_structure]).to(torch.float).sum(-1)
            mape = (torch.abs(card - pre_card) / card).mean()
            spm = spearman(pre_card, card)
            card_metrics[query_name_dict[query_structure]+" MAPE"] = mape
            card_metrics[query_name_dict[query_structure]+" Spearman"] = spm
        log_metrics('Card', card_metrics, writer)
    writer.write('\n')
    return all_metrics

def load_data(args, tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")
    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))
    
    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    return valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers

def main(args):
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataset_name = args.data_path.split('/')[1].split('-')[0]
    if args.data_path.split('/')[1].split('-')[1] == "237":
        dataset_name += "-237"
    filename = 'results/'+dataset_name+'_'+str(args.fraction)+'_'+str(args.thrshd)+'.txt'
    writer = open(filename, 'a+')

    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    
    global id2ent, id2rel
    with open('%s/id2ent.pkl'%args.data_path, 'rb') as f:
        id2ent = pickle.load(f)
    with open('%s/ent2id.pkl'%args.data_path, 'rb') as f:
        ent2id = pickle.load(f)
    with open('%s/id2rel.pkl'%args.data_path, 'rb') as f:
        id2rel = pickle.load(f)
    
    args.nentity = nentity
    args.nrelation = nrelation

    adj_list, edges_y, edges_p = read_triples([os.path.join(args.data_path, "train.txt")], args.nrelation, args.data_path)

    valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers = load_data(args, tasks)
    
    valid_queries = flatten_query(valid_queries)
    valid_dataloader = DataLoader(
        TestDataset(
            valid_queries, 
            args.nentity, 
            args.nrelation, 
        ), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn
    )

    test_queries = flatten_query(test_queries)
    test_dataloader = DataLoader(
        TestDataset(
            test_queries, 
            args.nentity, 
            args.nrelation, 
        ), 
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num, 
        collate_fn=TestDataset.collate_fn
    )
    
    model = KGReasoning(args, device, adj_list, query_name_dict, name_answer_dict)

    cp_thrshd = None
    if args.do_cp:
        cp_thrshd = get_cp_thrshd(model, valid_hard_answers, valid_easy_answers, args, valid_dataloader, query_name_dict, device)
    
    evaluate(model, test_hard_answers, test_easy_answers, args, test_dataloader, query_name_dict, device, writer, edges_y, edges_p, cp_thrshd)

if __name__ == '__main__':
    main(parse_args())