import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio import PDB
from scipy.spatial import distance_matrix
import gudhi
import copy
import sys
import os



class VLP:
    def __init__(self,pdb):
        self.pdb = pdb
        self.C_atom = []
        self.N_atom = []
        self.O_atom = []
        self.S_atom = []
        self.atom = []
        self.TopoFeature = []
        
        self.set_atom_coordinate()
        self.set_topo_feature()
    
    def set_atom_coordinate(self):
        # CA
        f = open('protein_C.pdb')
        contents = f.readlines()
        f.close()
        for line in contents:
            if line[0:4]=='ATOM':
                tmp = [float(line[30:38]),float(line[38:46]),float(line[46:54])]
                self.C_atom.append(tmp)
        
        # N
        f = open('protein_N.pdb')
        contents = f.readlines()
        f.close()
        for line in contents:
            if line[0:4]=='ATOM':
                tmp = [float(line[30:38]),float(line[38:46]),float(line[46:54])]
                self.N_atom.append(tmp)
        
        # O
        f = open('protein_O.pdb')
        contents = f.readlines()
        f.close()
        for line in contents:
            if line[0:4]=='ATOM':
                tmp = [float(line[30:38]),float(line[38:46]),float(line[46:54])]
                self.O_atom.append(tmp)
        
        # CA
        f = open('protein_S.pdb')
        contents = f.readlines()
        f.close()
        for line in contents:
            if line[0:4]=='ATOM':
                tmp = [float(line[30:38]),float(line[38:46]),float(line[46:54])]
                self.S_atom.append(tmp)
        
        self.atom = [ self.C_atom, self.N_atom, self.O_atom, self.C_atom+self.N_atom, self.C_atom+self.O_atom, self.N_atom+self.O_atom ]
        
        
    def set_rips_h0(self,Cut=12):
        def update_fea(m,idx,N):
            if idx>=N:
                m = m + 1
                return m
            else:
                temp = np.zeros((1,N))
                temp[0,:idx] = 1
                m = m + temp
                return m
        step = 0.5
        bin_num = int(10/step)
        # C,N,O,CN,CO,NO
        rips_h0 = [ np.zeros((1,bin_num)) ] * 6
        for i in range(6):
            atom_now = self.atom[i]
            rips_complex = gudhi.RipsComplex(points=atom_now, max_edge_length=Cut)
            PH = rips_complex.create_simplex_tree().persistence(min_persistence=0.1)
            for item in PH:
                if item[0]==0:
                    dea = item[1][1]
                    if dea==np.inf:
                        dea = 20 
                    idx = int(dea/step)
                    rips_h0[i] = update_fea(rips_h0[i],idx,bin_num)
        
        rips_h0_fea = []
        for i in range(6):
            rips_h0_fea = rips_h0_fea + rips_h0[i].tolist()[0]
        self.TopoFeature = self.TopoFeature + rips_h0_fea
    
    def set_rips_l0(self):
        def get_statistic(value):
            if len(value)==0:
                return [0]*7
            else:
                return [ np.max(value),np.min(value),np.sum(value),np.mean(value),np.std(value),np.var(value),len(value) ]
        fil = [ 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0]
        
        # C,N,O,CN,CO,NO
        rips_l0 = [ [] ]*6
        for i in range(6):
            atom = self.atom[i]
            dis_m = distance_matrix(atom, atom)
                
            for now_fil in fil:
                L = copy.deepcopy(dis_m)
                L[L<=now_fil] = -1
                L[L>0] = 0
                for k in range(len(dis_m)):
                    L[k,k] = - np.sum(L[k,:])
                eig = np.linalg.eigvalsh(L)
                eig = eig[eig>1e-8]
                    
                rips_l0[i] = rips_l0[i] + get_statistic(eig)
        
        rips_l0_fea = []
        for i in range(6):
            rips_l0_fea = rips_l0_fea + rips_l0[i]
        self.TopoFeature = self.TopoFeature + rips_l0_fea
    
    def set_alpha_h12(self):
        def sum_max_mean(ls):
            if len(ls)==0:
                return [0,0,0]
            else:
                return [ np.sum(ls),np.max(ls),np.mean(ls) ]
        def max_min(ls):
            if len(ls)==0:
                return [0,0]
            else:
                return [ np.max(ls),np.min(ls) ]
        
        alpha_h = [ ]
        # C,N,O,CN,CO,NO
        for i in range(6):
            atom = self.atom[i]
            alpha_complex = gudhi.AlphaComplex(points=atom)
            PH = alpha_complex.create_simplex_tree().persistence(min_persistence=0.1)
                
                
            birth1,death1,length1 = [],[],[]
            birth2,death2,length2 = [],[],[]
            for item in PH:
                        
                if item[0]==1:
                    birth1.append( item[1][0])
                    dea = item[1][1]
                    if dea==np.inf:
                        dea = 20
                    death1.append(dea)
                    length1.append(dea-item[1][0])
                elif item[0]==2:
                    birth2.append( item[1][0])
                    dea = item[1][1]
                    if dea==np.inf:
                        dea = 20
                    death2.append(dea)
                    length2.append(dea-item[1][0])
                    
            alpha_h.extend(sum_max_mean(length1))
            alpha_h.extend(max_min(birth1))
            alpha_h.extend(max_min(death1))
                
            alpha_h.extend(sum_max_mean(length2))
            alpha_h.extend(max_min(birth2))
            alpha_h.extend(max_min(death2))
        
        self.TopoFeature = self.TopoFeature + alpha_h
    
    def set_topo_feature(self):
        self.set_rips_h0()
        self.set_rips_l0()
        self.set_alpha_h12()



def prepare_structure(pdb,folder):
    # work directory
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.system('cp ../data/PDB/' +pdb + '.pdb ' + folder + 'prot.pdb')
    
    # extract protein coordinate
    tclfile = open(folder + 'vmd.tcl', 'w')
    tclfile.write('mol new {' + folder + 'prot.pdb} type {pdb} first 0 last 0 step 1 waitfor 1\n')
    tclfile.write('set protC [atomselect top "protein and name CA"]\n')
    tclfile.write('$protC writepdb ' + folder + 'protein_C.pdb\n')
    tclfile.write('$protC delete\n')
    
    tclfile.write('set protN [atomselect top "protein and element N"]\n')
    tclfile.write('$protN writepdb ' + folder + 'protein_N.pdb\n')
    tclfile.write('$protN delete\n')
    
    tclfile.write('set protO [atomselect top "protein and element O"]\n')
    tclfile.write('$protO writepdb ' + folder + 'protein_O.pdb\n')
    tclfile.write('$protO delete\n')
    
    tclfile.write('set protS [atomselect top "protein and element S"]\n')
    tclfile.write('$protS writepdb ' + folder + 'protein_S.pdb\n')
    tclfile.write('$protS delete\n')
    
    tclfile.write('exit')
    tclfile.close()
    
    os.system('vmd -dispdev text -e ' + folder + 'vmd.tcl')
    os.system('rm ' + folder + 'prot.pdb')
    os.system('rm ' + folder + 'vmd.tcl')


def get_old_data():
    filename = '../data/VLP200.csv'
    df = pd.read_csv(filename)
    data = []
    for i in range(200):
        pdb = df['PDB code'][i]
        seq = df['Protein sequence'][i]
        typ = df['Stoichiometry'][i]
        label = 1
        if typ==60:
            label = 0
        #print(i,pdb,label)
        data.append([ pdb,seq,label ])
    return data
    
        
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

def get_topo_feature(start,end):
    # 200 samples
    nowdata = get_old_data()
    
    # 706 samples
    #nowdata = get_new_data()
    
    for i in range(len(nowdata)):
        pdb,seq,label = nowdata[i]
        #pdb,label = nowdata[i]
        folder = './feature/' + pdb + '/'
        prepare_structure(pdb,folder)
        
        os.chdir(folder)
        
        VLP1 = VLP(pdb)
        topo_fea = VLP1.TopoFeature
        np.save('topo.npy',np.array(topo_fea))
        os.chdir('../..')
        print(i,pdb,'ok',len(topo_fea))
        
        

get_topo_feature()



