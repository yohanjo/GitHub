'''
Created on May 10, 2017

@author: Yohan
'''
import os, sys, re
from subprocess import Popen, PIPE
from ydata.csv_utils import *
from collections import defaultdict
import numpy as np

indir = "/Users/Yohan/Dropbox/Research/data/github/data"
outdir = "/Users/Yohan/Dropbox/Research/data/github/HSTTM"
jar_path = "/Users/Yohan/Downloads/stanford-parser-full-2016-10-31/stanford-parser.jar"
num_seqs = sys.maxint

csv.field_size_limit(sys.maxsize)

data = []
text_merged = []
cnt = 0
for filename in os.listdir(indir):
    if not filename.endswith(".csv"): continue
    time_data = dict()
    for row in iter_csv_header(indir+"/"+filename, readmode="rU"):
        try:
            text = row['text'].strip()
        except:
            print filename, row
            continue
        if len(text) == 0: continue
        time_data[row['time']] = (row['actor'], 
                                  re.sub("\\s+", " ", text))
    
    for inst_no, (_time, (user,text)) in \
                    enumerate(sorted(time_data.iteritems())):
        data.append((filename, inst_no, user, text))
        text_merged.append(text)
        
    cnt += 1
    if cnt >= num_seqs: break
      
# Tokenize  
p = Popen('java -cp '+jar_path+' edu.stanford.nlp.process.PTBTokenizer'+
          ' -preserveLines', shell=True, stdout=PIPE, stdin=PIPE)
tokenized_texts = p.communicate(input="\n".join(text_merged))[0]\
                   .strip().lower().split("\n")

# Print
seq_len = defaultdict(int)
inst_len = []
word_cnt = defaultdict(int)
with open(outdir+"/data.csv", "w") as f:
    outcsv = csv.writer(f)
    outcsv.writerow(['SeqId', 'InstNo', 'User', 'Text'])
    
    for (seq, inst, user, text), tokenized in zip(data, tokenized_texts):
        outcsv.writerow([seq, inst, user, tokenized])
        seq_len[seq] += 1
        inst_len.append(len(tokenized.split(" ")))
        for word in tokenized.split(" "):
            word_cnt[word] += 1
            
with open(outdir+"/word_count.csv", "w") as f:
    outcsv = csv.writer(f)
    outcsv.writerow(['Word', 'Count'])
    for word, cnt in sorted(word_cnt.iteritems(), key=lambda (k,v): v, 
                            reverse=True):
        outcsv.writerow([word, cnt])

with open(outdir+"/stats.txt", "w") as f:
    print>>f, "# seqs:", len(seq_len)
    print>>f, "# instances:", len(inst_len)
    print>>f, "# unique words:", len(word_cnt)
    print>>f, "# insts/seq: min=%d, median=%d, mean=%.2f, max=%d" % (
                        min(seq_len.values()), np.median(seq_len.values()), 
                        np.mean(seq_len.values()), max(seq_len.values()))
    print>>f, "# words/inst: min=%d, median=%d, mean=%.2f, max=%d" % (
                                        min(inst_len), np.median(inst_len), 
                                        np.mean(inst_len), max(inst_len))