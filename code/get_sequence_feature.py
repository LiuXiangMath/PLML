import os
import pandas as pd
import numpy as np
import sys
from Bio import SeqIO
from Bio import PDB
import random

start = int(sys.argv[1])
end = int(sys.argv[2])
seed = int(sys.argv[3])


def get_new_data():
    filename = '../data/VLP706.data'
    f = open(filename)
    tmp = f.readlines()
    f.close()
    data = []
    for item in tmp:
        pdb,label = item.strip().split(',')
        data.append([pdb,label])
    return data
    




import torch
import gc
device = 'cuda:2'
esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
esm_model = esm_model.to(device)




def get_esm_feature_remove(start,end,seed):
    def restrict_seq_len(fasta,pos,maxl=1300):
        if len(fasta)>maxl:
            half_window = maxl // 2
            start = max(0, pos - half_window)
            end = min(len(fasta), pos + half_window)
            if end - start < maxl:
                if start == 0:
                    end = min(len(fasta), start + maxl)
                else:
                    start = max(0, end - maxl)
            res = fasta[start:end]
            return res
        else:
            return fasta
    def get_seq(filename):
        seq = []
        for record in SeqIO.parse(filename, "fasta"):
            seq.append(str(record.seq))
        length = [ len(i) for i in seq ]
        index = length.index(np.max(length))
        return seq[index]

    def random_delete_amino_acids(sequence, ratio, seed=None):
        if seed is not None:
            random.seed(seed)
            
        seq_len = len(sequence)
        num_delete = max(1, int(seq_len * ratio))
    
        delete_indices = set(random.sample(range(seq_len), num_delete))
        
        new_sequence = ''.join([aa for i, aa in enumerate(sequence) if i not in delete_indices])
        return new_sequence

            
    
    # 706 samples
    nowdata = get_new_data()
    ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5  ]
    
    
    for i in range(start,end):
        pdb,label = nowdata[i]
        
        folder = './706-stability-feature/' + pdb + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.chdir(folder)
        
        batch_converter = alphabet.get_batch_converter()
        esm_model.eval()
        esm_feature = []
        
        # from fasta file
        filename = '../../../data/fasta-706/' + pdb + '.fasta'
        sequence = get_seq(filename)
        
        for ratio in ratios:
            fasta = random_delete_amino_acids(sequence, ratio, seed)
            num = len(fasta)//2
            fasta = restrict_seq_len(fasta,num)
            data = [ ('wild',fasta)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            
            with torch.no_grad():
                results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
            temp = results['representations'][33].cpu().numpy()
            temp = temp.reshape(-1,1280)
            esm_wild = np.mean(temp,axis=0).tolist()
            
            del batch_tokens, results, temp
            gc.collect()
            torch.cuda.empty_cache()
            trans_fea = esm_wild
            np.save('trans-ratio-' + str(int(100*ratio)) + '-seed-' + str(seed) + '.npy',np.array(trans_fea))
        print(i,folder,'ok')
        os.chdir('../..')









def get_esm_feature_add(start,end,seed):
    

    def restrict_seq_len(fasta,pos,maxl=1300):
        if len(fasta)>maxl:
            half_window = maxl // 2
            start = max(0, pos - half_window)
            end = min(len(fasta), pos + half_window)
            if end - start < maxl:
                if start == 0:
                    end = min(len(fasta), start + maxl)
                else:
                    start = max(0, end - maxl)
            res = fasta[start:end]
            return res
        else:
            return fasta
    def get_seq(filename):
        seq = []
        for record in SeqIO.parse(filename, "fasta"):
            seq.append(str(record.seq))
        length = [ len(i) for i in seq ]
        index = length.index(np.max(length))
        return seq[index]

    def random_delete_amino_acids(sequence, ratio, seed=None):
        if seed is not None:
            random.seed(seed)
            
        seq_len = len(sequence)
        num_delete = max(1, int(seq_len * ratio))
    
        delete_indices = set(random.sample(range(seq_len), num_delete))
        
        new_sequence = ''.join([aa for i, aa in enumerate(sequence) if i not in delete_indices])
        return new_sequence
    
    def random_add_amino_acids(sequence, ratio, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
        standard_aas = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
        seq_array = list(sequence)
        seq_len = len(seq_array)
        num_add = max(1, int(seq_len * ratio))
        
    
        insert_positions = np.random.randint(0, seq_len + 1, size=num_add)
        insert_aas = np.random.choice(standard_aas, size=num_add)
    
        sorted_indices = np.argsort(insert_positions)[::-1]
        for i in sorted_indices:
            seq_array.insert(insert_positions[i], insert_aas[i])
    
        return ''.join(seq_array)

            
    
    # 706 samples
    nowdata = get_new_data()
    ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5  ]
    
    
    for i in range(start,end):
        pdb,label = nowdata[i]
        
        folder = './706-stability-feature/' + pdb + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.chdir(folder)
        
        batch_converter = alphabet.get_batch_converter()
        esm_model.eval()
        esm_feature = []
        
        # from fasta file
        filename = '../../../data/fasta-706/' + pdb + '.fasta'
        sequence = get_seq(filename)
        
        for ratio in ratios:
            #fasta = random_delete_amino_acids(sequence, ratio, seed)
            fasta = random_add_amino_acids(sequence, ratio, seed)
            num = len(fasta)//2
            fasta = restrict_seq_len(fasta,num)
            data = [ ('wild',fasta)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            
            with torch.no_grad():
                results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
            temp = results['representations'][33].cpu().numpy()
            temp = temp.reshape(-1,1280)
            esm_wild = np.mean(temp,axis=0).tolist()
            
            del batch_tokens, results, temp
            gc.collect()
            torch.cuda.empty_cache()
            trans_fea = esm_wild
            np.save('add-trans-ratio-' + str(int(100*ratio)) + '-seed-' + str(seed) + '.npy',np.array(trans_fea))
        print(i,folder,'ok')
        os.chdir('../..')





def get_sem_feature_perturb(typ='remove',start,end,seed):
    if typ=='remove':
        get_esm_feature_remove(start,end,seed)
    elif typ=='add':
        get_esm_feature_add(start,end,seed)


get_esm_feature_perturb(start,end,seed)
        
    






        
        