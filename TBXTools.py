#    TBXTools
#    version: 2021/11/24
#    Copyright: Antoni Oliver (2021) - Universitat Oberta de Catalunya - aoliverg@uoc.edu
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import codecs
import sqlite3
import xml.etree.cElementTree as etree

import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.collocations import *
import re
import pickle
import gzip
import operator
import sys
import math

import string

import importlib

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy

try:
    import map_embeddings
except:
    pass
try:
    import embeddings
except:
    pass

class TBXTools:
    '''Class for automatic terminology extraction and terminology management.'''
    def __init__(self):
        self.maxinserts=10000 #controls the maximum number of inserts in memory
        self.sl_lang=""
        self.tl_lang=""
        self.max_id_corpus=0
        
        self.sl_stopwords=[]
        self.tl_stopwords=[]
        self.sl_inner_stopwords=[]
        self.tl_inner_stopwords=[]
        self.sl_exclsions_regexps=[]
        self.tl_exclusion_regexps=[]
        self.sl_morphonorm_rules=[]
        self.tl_morphonorm_rules=[]
        self.evaluation_terms={}
        self.tsr_terms=[]
        self.exclusion_terms={}
        self.exclusion_no_terms={}
        self.ngrams={}
        self.tagged_ngrams={}
        self.term_candidates={}
        self.linguistic_patterns={}
        
        self.knownterms=[]
        self.n_min=1
        self.n_max=5
        
        self.punctuation=string.punctuation
        self.sl_stopwords.extend(self.punctuation)
        self.tl_stopwords.extend(self.punctuation)
        self.sl_inner_stopwords.extend(self.punctuation)
        self.tl_inner_stopwords.extend(self.punctuation)
        
        self.specificSLtokenizer=False
        self.specificTLtokenizer=False
        
        self.SLtokenizer=None
        self.TLtokenizer=None
        
        
    def create_project(self,project_name,sl_lang,tl_lang="null",overwrite=False):
        '''Opens a project. If the project already exists, it raises an exception. To avoid the exception use overwrite=True. To open existing projects, use the open_project method.'''
        if os.path.isfile(project_name) and not overwrite:
                raise Exception("This file already exists")
        
        else:
            if os.path.isfile(project_name) and overwrite:
                os.remove(project_name)
            self.sl_lang=sl_lang
            self.tl_lang=tl_lang
            self.conn=sqlite3.connect(project_name)
            self.cur = self.conn.cursor() 
            self.cur2 = self.conn.cursor()
            with self.conn:
                self.cur = self.conn.cursor()
                self.cur.execute("CREATE TABLE configuration(id INTEGER PRIMARY KEY, sl_lang TEXT, tl_lang TEXT)")
                self.cur.execute("CREATE TABLE sl_corpus(id INTEGER PRIMARY KEY, segment TEXT)")
                self.cur.execute("CREATE TABLE tl_corpus(id INTEGER PRIMARY KEY, segment TEXT)")
                self.cur.execute("CREATE TABLE sl_tagged_corpus(id INTEGER PRIMARY KEY, tagged_segment TEXT)")
                self.cur.execute("CREATE TABLE tl_tagged_corpus(id INTEGER PRIMARY KEY, tagged_segment TEXT)")
                self.cur.execute("CREATE TABLE sl_stopwords (id INTEGER PRIMARY KEY AUTOINCREMENT, sl_stopword TEXT)")
                self.cur.execute("CREATE TABLE sl_inner_stopwords (id INTEGER PRIMARY KEY AUTOINCREMENT, sl_inner_stopword TEXT)")
                self.cur.execute("CREATE TABLE tl_stopwords (id INTEGER PRIMARY KEY AUTOINCREMENT, tl_stopword TEXT)")
                self.cur.execute("CREATE TABLE tl_inner_stopwords (id INTEGER PRIMARY KEY AUTOINCREMENT, tl_inner_stopword TEXT)")
                self.cur.execute("CREATE TABLE sl_exclusion_regexps (id INTEGER PRIMARY KEY AUTOINCREMENT, sl_exclusion_regexp TEXT)")
                self.cur.execute("CREATE TABLE tl_exclusion_regexps (id INTEGER PRIMARY KEY AUTOINCREMENT, tl_exclusion_regexp TEXT)")
                self.cur.execute("CREATE TABLE sl_morphonorm_rules (id INTEGER PRIMARY KEY AUTOINCREMENT, sl_morphonorm_rule TEXT)")
                self.cur.execute("CREATE TABLE tl_morphonorm_rules (id INTEGER PRIMARY KEY AUTOINCREMENT, tl_morphonorm_rule TEXT)")
                self.cur.execute("CREATE TABLE sl_patterns (id INTEGER PRIMARY KEY AUTOINCREMENT, sl_pattern TEXT)")
                self.cur.execute("CREATE TABLE tl_patterns (id INTEGER PRIMARY KEY AUTOINCREMENT, tl_pattern TEXT)")
                self.cur.execute("CREATE TABLE evaluation_terms (id INTEGER PRIMARY KEY AUTOINCREMENT, sl_term TEXT, tl_term TEXT)")
                self.cur.execute("CREATE TABLE reference_terms (id INTEGER PRIMARY KEY AUTOINCREMENT, sl_term TEXT, tl_term TEXT)")
                self.cur.execute("CREATE TABLE compoundify_terms_sl (id INTEGER PRIMARY KEY AUTOINCREMENT, term TEXT)")
                self.cur.execute("CREATE TABLE compoundify_terms_tl (id INTEGER PRIMARY KEY AUTOINCREMENT, term TEXT)")
                self.cur.execute("CREATE TABLE tsr_terms (id INTEGER PRIMARY KEY AUTOINCREMENT, term TEXT)")
                self.cur.execute("CREATE TABLE exclusion_terms (id INTEGER PRIMARY KEY AUTOINCREMENT, sl_term TEXT, tl_term TEXT)")
                self.cur.execute("CREATE TABLE exclusion_noterms (id INTEGER PRIMARY KEY AUTOINCREMENT, sl_term TEXT, tl_term TEXT)")
                self.cur.execute("CREATE TABLE tokens (id INTEGER PRIMARY KEY AUTOINCREMENT, token TEXT, frequency INTEGER)")
                self.cur.execute("CREATE TABLE ngrams (id INTEGER PRIMARY KEY AUTOINCREMENT, ngram TEXT, n INTEGER, frequency INTEGER)")
                self.cur.execute("CREATE TABLE tagged_ngrams (id INTEGER PRIMARY KEY AUTOINCREMENT, ngram TEXT, tagged_ngram TEXT, n INTEGER, frequency INTEGER)")
                
                self.cur.execute("CREATE INDEX indextaggedngram on tagged_ngrams (ngram);")
                
                self.cur.execute("CREATE TABLE embeddings_sl (id INTEGER PRIMARY KEY AUTOINCREMENT, candidate TEXT, embedding BLOB)")
                self.cur.execute("CREATE INDEX indexembeddings_sl on embeddings_sl (candidate);")
                
                self.cur.execute("CREATE TABLE embeddings_sl_ref (id INTEGER PRIMARY KEY AUTOINCREMENT, candidate TEXT, embedding BLOB)")
                self.cur.execute("CREATE INDEX indexembeddings_sl_ref on embeddings_sl_ref (candidate);")
                
                self.cur.execute("CREATE TABLE embeddings_tl (id INTEGER PRIMARY KEY AUTOINCREMENT, candidate TEXT, embedding BLOB)")
                self.cur.execute("CREATE INDEX indexembeddings_tl on embeddings_tl (candidate);")
                
                self.cur.execute("CREATE TABLE term_candidates (id INTEGER PRIMARY KEY AUTOINCREMENT, candidate TEXT, n INTEGER, frequency INTEGER, measure TEXT, value FLOAT)")
                self.cur.execute("CREATE TABLE index_pt(id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, target TEXT, probability FLOAT)")
                self.cur.execute("CREATE INDEX index_index_pt on index_pt (source);")
                self.cur.execute("CREATE TABLE linguistic_patterns (id INTEGER PRIMARY KEY AUTOINCREMENT, linguistic_pattern TEXT)")
                
                self.cur.execute("INSERT INTO configuration (id, sl_lang, tl_lang) VALUES (?,?,?)",[0,self.sl_lang,self.tl_lang])
                self.conn.commit()
                
    def open_project(self,project_name):
        '''Opens an existing project. If the project doesn't exist it raises an exception.'''
        if not os.path.isfile(project_name):
                raise Exception("Project not found")
        else:
            self.conn=sqlite3.connect(project_name)
            self.cur = self.conn.cursor() 
            self.cur2 = self.conn.cursor()
            self.maxsc=0
            self.maxtc=0
            with self.conn:
                self.cur.execute('SELECT sl_lang from configuration where id=0')
                self.sl_lang=self.cur.fetchone()[0]
                self.cur.execute('SELECT tl_lang from configuration where id=0')
                self.tl_lang=self.cur.fetchone()[0]
                #Maximum position in corpora
                self.maxpositions=[]
                self.cur.execute('SELECT max(id) from sl_corpus')
                self.qresult=self.cur.fetchone()[0]
                if not self.qresult==None:
                    self.maxpositions.append(self.qresult)
                else:
                    self.maxpositions.append(0)
                self.cur.execute('SELECT max(id) from tl_corpus')
                self.qresult=self.cur.fetchone()[0]
                if not self.qresult==None:
                    self.maxpositions.append(self.qresult)
                else:
                    self.maxpositions.append(0)
                self.cur.execute('SELECT max(id) from sl_tagged_corpus')
                self.qresult=self.cur.fetchone()[0]
                if not self.qresult==None:
                    self.maxpositions.append(self.qresult)
                else:
                    self.maxpositions.append(0)
                self.cur.execute('SELECT max(id) from tl_tagged_corpus')
                self.qresult=self.cur.fetchone()[0]
                if not self.qresult==None:
                    self.maxpositions.append(self.qresult)
                else:
                    self.maxpositions.append(0)
                self.max_id_corpus=max(self.maxpositions)

                
                #loading of stopwords
                self.cur.execute('SELECT sl_stopword from sl_stopwords')
                self.data=self.cur.fetchall()
                for self.d in self.data:
                    self.sl_stopwords.append(self.d[0])

                self.cur.execute('SELECT tl_stopword from tl_stopwords')
                self.data=self.cur.fetchall()
                for self.d in self.data:
                    self.tl_stopwords.append(self.d[0])

                self.cur.execute('SELECT sl_inner_stopword from sl_inner_stopwords')
                self.data=self.cur.fetchall()
                for self.d in self.data:
                    self.sl_inner_stopwords.append(self.d[0])

                self.cur.execute('SELECT tl_inner_stopword from tl_inner_stopwords')
                self.data=self.cur.fetchall()
                for self.d in self.data:
                    self.tl_inner_stopwords.append(self.d[0])
                    
                self.cur.execute('SELECT sl_exclusion_regexp from sl_exclusion_regexps')
                self.data=self.cur.fetchall()
                for self.d in self.data:
                    self.sl_exclusion_regexps.append(self.d[0])
                    
                self.cur.execute('SELECT tl_exclusion_regexp from tl_exclusion_regexps')
                self.data=self.cur.fetchall()
                for self.d in self.data:
                    self.tl_exclusion_regexps.append(self.d[0])
                    
                self.cur.execute('SELECT sl_morphonorm_rule from sl_morphonorm_rules')
                self.data=self.cur.fetchall()
                for self.d in self.data:
                    self.sl_morphonorm_rules.append(self.d[0])
                
                self.cur.execute('SELECT tl_morphonorm_rule from tl_morphonorm_rules')
                self.data=self.cur.fetchall()
                for self.d in self.data:
                    self.tl_morphonorm_rules.append(self.d[0])
                    
                self.cur.execute('SELECT sl_term,tl_term from evaluation_terms')
                self.data=self.cur.fetchall()
                for self.d in self.data:
                    self.evaluation_terms[self.d[0]]=self.d[1]

    #METODES DELETES
    def delete_sl_corpus(self):
        '''Deletes de source language corpus.'''
        with self.conn:
            self.cur.execute('DELETE FROM sl_corpus')
            self.conn.commit()
    
    def delete_tl_corpus(self):
        '''Deletes de target language corpus.'''
        with self.conn:
            self.cur.execute('DELETE FROM tl_corpus')
            self.conn.commit()
    
    def delete_sl_tagged_corpus(self):
        '''Deletes the source language tagged corpus.'''
        with self.conn:
            self.cur.execute('DELETE FROM sl_tagged_corpus')
            self.conn.commit()
    
    def delete_tl_tagged_corpus(self):
        '''Deletes the target language tagged corpu.'''
        with self.conn:
            self.cur.execute('DELETE FROM tl_tagged_corpus')
            self.conn.commit()
    
    def delete_sl_stopwords(self):
        '''Deletes the stop-words for the source language.'''
        self.sl_stopwords=[]
        with self.conn:
            self.cur.execute('DELETE FROM sl_stopwords')
            self.conn.commit()
            
    def delete_tl_stopwords(self):
        '''Deletes the stop-words fot the target language.'''
        self.tl_stopwords=[]
        with self.conn:
            self.cur.execute('DELETE FROM tl_stopwords')
            self.conn.commit()
            
    def delete_sl_inner_stopwords(self):
        '''Deletes the inner stop-words for the source language.'''
        self.sl_inner_stopwords=[]
        with self.conn:
            self.cur.execute('DELETE FROM sl_inner_stopwords')
            self.conn.commit()
            
    def delete_tl_inner_stopwords(self):
        '''Deletes the innter stop-words for the target language.'''
        self.tl_inner_stopwords=[]
        with self.conn:
            self.cur.execute('DELETE FROM tl_inner_stopwords')
            self.conn.commit()
    
    def delete_sl_exclusion_regexps(self):
        '''Deletes the exclusion regular expressions for the source language.'''
        self.sl_exclusion_regexps=[]
        with self.conn:
            self.cur.execute('DELETE FROM sl_exclusion_regexps')
            self.conn.commit()
    
    def delete_tl_exclusion_regexps(self):
        '''Deletes the exclusion regular expressions for the target language.'''
        self.tl_exclusion_regexps=[]
        with self.conn:
            self.cur.execute('DELETE FROM tl_exclusion_regexps')
            self.conn.commit()
            
    def delete_sl_morphonorm_rules(self):
        '''Deletes the morphological normalisation rules for the source language.'''
        self.sl_morphonorm_rules=[]
        with self.conn:
            self.cur.execute('DELETE FROM sl_morphonorm_rules')
            self.conn.commit()
            
    def delete_tl_morphonorm_rules(self):
        '''Deletes the morphological normalisation rules for the target language.'''
        self.tl_morphonorm_rules=[]
        with self.conn:
            self.cur.execute('DELETE FROM tl_morphonorm_rules')
            self.conn.commit()

    def delete_evaluation_terms(self):
        '''Deletes the evaluation terms.'''
        self.evaluation_terms={}
        with self.conn:
            self.cur.execute('DELETE FROM evaluation_terms')
            self.conn.commit()
            
    def delete_exclusion_terms(self):
        '''Deletes the exclusion terms.'''
        self.exclusion_terms={}
        with self.conn:
            self.cur.execute('DELETE FROM exclusion_terms')
            self.conn.commit()   
            
    def delete_exclusion_no_terms(self):
        '''Deletes the exclusion no terms.'''
        self.exclusion_terms={}
        with self.conn:
            self.cur.execute('DELETE FROM exclusion_terms')
            self.conn.commit() 
            
    def delete_linguistic_patterns(self):
        '''Deletes the linguistic patterns for linguistic terminology extraction.'''
        self.exclusion_terms={}
        with self.conn:
            self.cur.execute('DELETE FROM exclusion_terms')
            self.conn.commit() 
    
    def delete_tokens(self):
        '''Deletes the tokens,'''
        self.ngrams={}
        with self.conn:
            self.cur.execute('DELETE FROM tokens')
            self.conn.commit()
            
    def delete_ngrams(self):
        '''Deletes the ngrams.'''
        self.ngrams={}
        with self.conn:
            self.cur.execute('DELETE FROM ngrams')
            self.conn.commit()
            
    def delete_tagged_ngrams(self):
        '''Deletes the tagged ngrams.'''
        self.tagged_ngrams={}
        with self.conn:
            self.cur.execute('DELETE FROM tagged_ngrams')
            self.conn.commit()
    
    def delete_term_candidates(self):
        '''Deletes the term candidates.'''
        self.term_candidates={}
        with self.conn:
            self.cur.execute('DELETE FROM term_candidates')
            self.conn.commit()
                
    
    def load_sl_corpus(self,corpusfile, encoding="utf-8", compoundify=False):
        '''Loads a monolingual corpus for the source language. It's recommended, but not compulsory, that the corpus is segmented (one segment per line). Use TBXTools external tools to segment the corpus. A plain text corpus (not segmented), can be aslo used.'''
        if compoundify:
            compterms=[]
            self.cur.execute('SELECT term from compoundify_terms_sl')
            data=self.cur.fetchall()
            for d in data:
                compterms.append(d[0])
            
        self.cf=codecs.open(corpusfile,"r",encoding=encoding,errors="ignore")
        self.data=[]
        self.continserts=0
        for self.line in self.cf:
            
            self.continserts+=1
            self.max_id_corpus+=1
            self.record=[]
            self.line=self.line.rstrip()
            if compoundify:
                for compterm in compterms:
                    if self.line.find(compterm)>=1:
                        comptermMOD=compterm.replace(" ","▁")
                        self.line=self.line.replace(compterm,comptermMOD)
            self.record.append(self.max_id_corpus)
            self.record.append(self.line)
            self.data.append(self.record)
            if self.continserts==self.maxinserts:
                self.cur.executemany("INSERT INTO sl_corpus (id, segment) VALUES (?,?)",self.data)
                self.data=[]
                self.continserts=0
        with self.conn:
            self.cur.executemany("INSERT INTO sl_corpus (id, segment) VALUES (?,?)",self.data)    
        self.conn.commit()
        
    def load_tl_corpus(self,corpusfile, encoding="utf-8", compoundify=False):
        '''Loads a monolingual corpus for the target language. It's recommended, but not compulsory, that the corpus is segmented (one segment per line). Use TBXTools external tools to segment the corpus. A plain text corpus (not segmented), can be aslo used.'''
        if compoundify:
            compterms=[]
            self.cur.execute('SELECT term from compoundify_terms_tl')
            data=self.cur.fetchall()
            for d in data:
                compterms.append(d[0])
        self.cf=codecs.open(corpusfile,"r",encoding=encoding,errors="ignore")
        self.data=[]
        self.continserts=0
        for self.line in self.cf:
            
            self.continserts+=1
            self.max_id_corpus+=1
            self.record=[]
            self.line=self.line.rstrip()
            if compoundify:
                for compterm in compterms:
                    if self.line.find(compterm)>=1:
                        comptermMOD=compterm.replace(" ","▁")
                        self.line=self.line.replace(compterm,comptermMOD)
            self.record.append(self.max_id_corpus)
            self.record.append(self.line)
            self.data.append(self.record)
            if self.continserts==self.maxinserts:
                self.cur.executemany("INSERT INTO tl_corpus (id, segment) VALUES (?,?)",self.data)
                self.data=[]
                self.continserts=0
        with self.conn:
            self.cur.executemany("INSERT INTO tl_corpus (id, segment) VALUES (?,?)",self.data)    
        self.conn.commit()
    
    def load_sl_corpus_ref(self,corpusfile, encoding="utf-8", compoundify=False):
        '''Loads a monolingual corpus for the source language. It's recommended, but not compulsory, that the corpus is segmented (one segment per line). Use TBXTools external tools to segment the corpus. A plain text corpus (not segmented), can be aslo used.'''
        if compoundify:
            compterms=[]
            self.cur.execute('SELECT term from compoundify_terms_sl')
            data=self.cur.fetchall()
            for d in data:
                compterms.append(d[0])
        self.cf=codecs.open(corpusfile,"r",encoding=encoding,errors="ignore")
        self.data=[]
        self.continserts=0
        for self.line in self.cf:
            
            self.continserts+=1
            self.max_id_corpus+=1
            self.record=[]
            self.line=self.line.rstrip()
            if compoundify:
                for compterm in compterms:
                    if self.line.find(compterm)>=1:
                        comptermMOD=compterm.replace(" ","▁")
                        self.line=self.line.replace(compterm,comptermMOD)
            self.record.append(self.max_id_corpus)
            self.record.append(self.line)
            self.data.append(self.record)
            if self.continserts==self.maxinserts:
                self.cur.executemany("INSERT INTO sl_corpus_ref (id, segment) VALUES (?,?)",self.data)
                self.data=[]
                self.continserts=0
        with self.conn:
            self.cur.executemany("INSERT INTO sl_corpus_ref (id, segment) VALUES (?,?)",self.data)    
        self.conn.commit()
        
    def load_sl_tl_corpus(self,slcorpusfile, tlcorpusfile, encoding="utf-8"):
        '''Loads a bilingual corpus in Moses format (that is, in two independent files. It expects one segment per line.'''
        self.slcf=codecs.open(slcorpusfile,"r",encoding=encoding)
        self.tlcf=codecs.open(tlcorpusfile,"r",encoding=encoding)
        self.sl_data=[]
        self.tl_data=[]
        self.continserts=0
        while 1:
            self.sl_segment=self.slcf.readline()
            if not self.sl_segment:
                break
            self.tl_segment=self.tlcf.readline()
            self.continserts+=1
            self.max_id_corpus+=1
            self.sl_record=[]
            self.tl_record=[]
            self.sl_segment=self.sl_segment.rstrip()
            self.tl_segment=self.tl_segment.rstrip()
            
            self.sl_record.append(self.max_id_corpus)
            self.tl_record.append(self.max_id_corpus)
            self.sl_record.append(self.sl_segment)
            self.tl_record.append(self.tl_segment)
            self.sl_data.append(self.sl_record)
            self.tl_data.append(self.tl_record)
            if self.continserts==self.maxinserts:
                self.cur.executemany("INSERT INTO sl_corpus (id, segment) VALUES (?,?)",self.sl_data)
                self.cur.executemany("INSERT INTO tl_corpus (id, segment) VALUES (?,?)",self.tl_data)
                self.sl_data=[]
                self.tl_data=[]
                self.continserts=0
        with self.conn:
            self.cur.executemany("INSERT INTO sl_corpus (id, segment) VALUES (?,?)",self.sl_data)   
            self.cur.executemany("INSERT INTO tl_corpus (id, segment) VALUES (?,?)",self.tl_data)  
        self.conn.commit()
        
    def load_parallel_corpus(self,parallelcorpusfile, reverse=False, encoding="utf-8"):
        '''Loads a bilingual corpus in Moses format (that is, in two independent files. It expects one segment per line.'''
        slcf=codecs.open(parallelcorpusfile,"r",encoding=encoding)
        sl_data=[]
        tl_data=[]
        continserts=0
        while 1:
            segment=slcf.readline()
            if not segment:
                break            
            continserts+=1
            self.max_id_corpus+=1
            sl_record=[]
            tl_record=[]
            sl_segment=segment.rstrip().split("\t")[0]
            tl_segment=segment.rstrip().split("\t")[1]
            
            sl_record.append(self.max_id_corpus)
            tl_record.append(self.max_id_corpus)
            sl_record.append(sl_segment)
            tl_record.append(tl_segment)
            sl_data.append(sl_record)
            tl_data.append(tl_record)
            if continserts==self.maxinserts:
                self.cur.executemany("INSERT INTO sl_corpus (id, segment) VALUES (?,?)",sl_data)
                self.cur.executemany("INSERT INTO tl_corpus (id, segment) VALUES (?,?)",tl_data)
                sl_data=[]
                tl_data=[]
                continserts=0
        with self.conn:
            self.cur.executemany("INSERT INTO sl_corpus (id, segment) VALUES (?,?)",sl_data)   
            self.cur.executemany("INSERT INTO tl_corpus (id, segment) VALUES (?,?)",tl_data)  
        self.conn.commit()
                    
    def load_tmx(self,tmx_file, sl_code=None, tl_code= None):
        '''Loads a parallel corpus from a TMX file. Source and target language codes should be given. The codes must be the exactly the same as in the TMX file. In no codes are provided, the codes used in the TBXTools project will be used.'''
        self.continserts=0
        if sl_code==None:
            self.sl_code=self.sl_lang
        else:
            self.sl_code=sl_code
        if tl_code==None:
            self.tl_code=self.tl_lang
        else:
            self.tl_code=tl_code
        self.data1=[]
        self.data2=[]
        self.sl_segment=""
        self.tl_segment=""
        self.current_lang=""
        for self.event, self.elem in etree.iterparse(tmx_file,events=("start","end")):
            if self.event=='start':
                if self.elem.tag=="tu" and not self.sl_segment=="" and not self.tl_segment=="":
                    self.continserts+=1
                    self.max_id_corpus+=1
                    self.record1=[]
                    self.record2=[]
                    self.record1.append(self.max_id_corpus)
                    self.record1.append(self.sl_segment)
                    self.data1.append(self.record1)
                    
                    self.record2.append(self.max_id_corpus)
                    self.record2.append(self.tl_segment)
                    self.data2.append(self.record2)
                    self.sl_segment=""
                    self.tl_segment=""
                    if self.continserts==self.maxinserts:
                        self.cur.executemany("INSERT INTO sl_corpus (id, segment) VALUES (?,?)",self.data1) 
                        self.cur.executemany("INSERT INTO tl_corpus (id, segment) VALUES (?,?)",self.data2)
                        self.data1=[]
                        self.data2=[]
                        self.continserts=0
                        self.conn.commit()
                elif self.elem.tag=="tuv":
                    self.current_lang=self.elem.attrib['{http://www.w3.org/XML/1998/namespace}lang']
                elif self.elem.tag=="seg":
                    if self.elem.text:
                        self.segmentext=self.elem.text
                    else:
                        self.segmentext=""
                    if self.current_lang==sl_code:
                        self.sl_segment=self.segmentext
                    if self.current_lang==tl_code:
                        self.tl_segment=self.segmentext                
                        

    def load_sl_tagged_corpus(self,corpusfile,format="TBXTools",encoding="utf-8"):
        '''Loads the source language tagged corpus. 3 formats are allowed:
        - TBXTools: The internal format used by TBXTools. One tagged segment per line.
                f1|l1|t1|p1 f2|l2|t2|p2 ... fn|ln|tn|pn 
        - Freeling: One token per line and segments separated by blank lines
                f1 l1 t1 p1
                f2 l2 t2 p2
                ...
                fn ln tn pn
        - Conll: One of the output formats guiven by the Standford Core NLP analyzer.  On token per line and segments separated by blank lines
                id1 f1 l1 t1 ...
                id2 f2 l2 t2 ...
                ...
                idn fn ln tn ...
        '''
        self.validformarts=["TBXTools","freeling","conll"]
        #TODO: Raise exception if not a valid format.
        self.cf=codecs.open(corpusfile,"r",encoding=encoding)
        if format=="TBXTools":
            
            self.data=[]
            self.continserts=0
            for self.line in self.cf:
                self.continserts+=1
                self.max_id_corpus+=1
                self.record=[]
                self.line=self.line.rstrip()
                self.record.append(self.max_id_corpus)
                self.record.append(self.line)
                self.data.append(self.record)
                if self.continserts==self.maxinserts:
                    self.cur.executemany("INSERT INTO sl_tagged_corpus (id, tagged_segment) VALUES (?,?)",self.data)
                    self.data=[]
                    self.continserts=0
                    
            with self.conn:
                self.cur.executemany("INSERT INTO sl_tagged_corpus (id, tagged_segment) VALUES (?,?)",self.data)    
            self.conn.commit()
        elif format=="freeling":
            self.data=[]
            self.continserts=0
            self.segment=[]
            for self.line in self.cf:
                self.line=self.line.rstrip()
                if self.line=="":
                    self.continserts+=1
                    self.max_id_corpus+=1
                    self.record=[]
                    self.record.append(self.max_id_corpus)
                    self.record.append(" ".join(self.segment))
                    print(" ".join(self.segment))
                    self.data.append(self.record)
                    if self.continserts==self.maxinserts:
                        self.cur.executemany("INSERT INTO sl_tagged_corpus (id, tagged_segment) VALUES (?,?)",self.data)
                        self.data=[]
                        self.continserts=0
                        self.data=[]
                        self.conn.commit()
                    self.segment=[]
                        
                else:
                    self.camps=self.line.split(" ")
                    self.token=self.camps[0]+"|"+self.camps[1]+"|"+self.camps[2]

                    self.segment.append(self.token)
            with self.conn:
                self.cur.executemany("INSERT INTO sl_tagged_corpus (id, tagged_segment) VALUES (?,?)",self.data)    
                self.conn.commit()
                    
        
                    
        elif format=="conll":
            self.data=[]
            self.continserts=0
            self.segment=[]
            for self.line in self.cf:
                self.line=self.line.rstrip()
                if self.line=="":
                    self.continserts+=1
                    self.max_id_corpus+=1
                    self.record=[]
                    self.record.append(self.max_id_corpus)
                    self.record.append(" ".join(self.segment))
                    self.data.append(self.record)
                    if self.continserts==self.maxinserts:
                        self.cur.executemany("INSERT INTO sl_tagged_corpus (id, tagged_segment) VALUES (?,?)",self.data)
                        self.data=[]
                        self.continserts=0
                        self.data=[]
                        self.conn.commit()
                    self.segment=[]
                        
                else:
                    self.camps=self.line.split("\t")
                    self.token=self.camps[1]+"|"+self.camps[2]+"|"+self.camps[3]
                    self.segment.append(self.token)
                
                
                
            with self.conn:
                self.cur.executemany("INSERT INTO sl_tagged_corpus (id, tagged_segment) VALUES (?,?)",self.data)    
            self.conn.commit()
            
            
        


    def load_sl_stopwords(self,fitxer,encoding="utf-8"):
        '''Loads the stopwords for the source language.'''
        self.fc=codecs.open(fitxer,"r",encoding)
        self.data=[]
        self.record=[]
        while 1:
            self.linia=self.fc.readline()
            if not self.linia:
                break 
            self.linia=self.linia.rstrip()
            self.sl_stopwords.append(self.linia)
            self.record.append(self.linia)
            self.data.append(self.record)
            self.record=[]
        
        for self.punct in self.punctuation:
            self.record.append(self.punct)
            self.data.append(self.record)
            self.record=[]
        with self.conn:
            self.cur.executemany("INSERT INTO sl_stopwords (sl_stopword) VALUES (?)",self.data)  
            
    def load_tl_stopwords(self,fitxer,encoding="utf-8"):
        '''Loads the stopwords for the target language.'''
        self.fc=codecs.open(fitxer,"r",encoding)
        self.data=[]
        self.record=[]
        while 1:
            self.linia=self.fc.readline()
            if not self.linia:
                break 
            self.linia=self.linia.rstrip()
            self.tl_stopwords.append(self.linia)
            self.record.append(self.linia)
            self.data.append(self.record)
            self.record=[]
        
        for self.punct in self.punctuation:
            self.record.append(self.punct)
            self.data.append(self.record)
            self.record=[]
        with self.conn:
            self.cur.executemany("INSERT INTO tl_stopwords (tl_stopword) VALUES (?)",self.data) 

    def load_sl_inner_stopwords(self,fitxer,encoding="utf-8"):
        '''Loads the stopwords for the source language.'''
        self.fc=codecs.open(fitxer,"r",encoding)
        self.data=[]
        self.record=[]
        while 1:
            self.linia=self.fc.readline()
            if not self.linia:
                break 
            self.linia=self.linia.rstrip()
            self.sl_inner_stopwords.append(self.linia)
            self.record.append(self.linia)
            self.data.append(self.record)
            self.record=[]
        
        for self.punct in self.punctuation:
            self.record.append(self.punct)
            self.data.append(self.record)
            self.record=[]
        with self.conn:
            self.cur.executemany("INSERT INTO sl_inner_stopwords (sl_inner_stopword) VALUES (?)",self.data)  
            
    def load_tl_inner_stopwords(self,fitxer,encoding="utf-8"):
        '''Loads the inner stopwords for the target language.'''
        self.fc=codecs.open(fitxer,"r",encoding)
        self.data=[]
        self.record=[]
        while 1:
            self.linia=self.fc.readline()
            if not self.linia:
                break 
            self.linia=self.linia.rstrip()
            self.tl_inner_stopwords.append(self.linia)
            self.record.append(self.linia)
            self.data.append(self.record)
            self.record=[]
        
        for self.punct in self.punctuation:
            self.record.append(self.punct)
            self.data.append(self.record)
            self.record=[]
        with self.conn:
            self.cur.executemany("INSERT INTO tl_inner_stopwords (tl_inner_stopword) VALUES (?)",self.data)  

    def load_evaluation_terms(self,arxiu,encoding="utf-8",nmin=0,nmax=1000):
        '''Loads the evaluation terms from a tabulated text'''
        #TODO Load from TBX
        
        self.cf=codecs.open(arxiu,"r",encoding=encoding)
        self.data=[]
        self.continserts=0
        for self.line in self.cf:
            
            self.line=self.line.rstrip()
            self.continserts+=1            
            self.record=[]
            self.line=self.line.rstrip()
            self.camps=self.line.split("\t")
            if len(self.camps)==1:
                if len(self.camps[0].split(" "))>=nmin and len(self.camps[0].split(" "))<=nmax:
                    
                    self.record.append(self.camps[0])
                    self.record.append("_")
                    self.evaluation_terms[self.camps[0]]="_"
                    self.data.append(self.record)
            elif len(self.camps)>1:
                if len(self.camps[0].split(" "))>=nmin and len(self.camps[0].split(" "))<=nmax:
                    self.record.append(self.camps[0])
                    self.record.append(self.camps[1])
                    self.evaluation_terms[self.camps[0]]=self.camps[1]
                    self.data.append(self.record)
            if self.continserts==self.maxinserts:
                
                self.cur.executemany("INSERT INTO evaluation_terms (sl_term,tl_term) VALUES (?,?)",self.data)
                self.data=[]
                self.continserts=0
        with self.conn:
            self.cur.executemany("INSERT INTO evaluation_terms (sl_term,tl_term) VALUES (?,?)",self.data)
    
        self.conn.commit()
    
    def namespace(self,element):
        m = re.match(r'\{.*\}', element.tag)
        return m.group(0) if m else ''
        
    def find_translation_reference_terms(self,term):
        self.cur.execute("SELECT tl_term FROM reference_terms where sl_term='"+str(term)+"'")
        tlterms=[]
        for self.s in self.cur.fetchall():
            tlterms.append(self.s[0])
        if len(tlterms)>0:
            return(", ".join(tlterms))
        else:
            return(None)
        
    
    def load_reference_terms(self,arxiu,sl="en",tl="es",format="TBX",encoding="utf-8"):
        '''Loads the reference terms from a TBX or tab txt file'''
        if format=="TBX":
            data=[]
            slterm=[]
            tlterm=[]
            lang=""
            for event, elem in etree.iterparse(arxiu,events=("start", "end")):
                tag=elem.tag.replace(self.namespace(elem),"")
                if event=="end" and tag in ["conceptEntry","termEntry"]:
                    if len(slterm)>0 and len(tlterm)>0:
                        record=[]
                        for slt in slterm:
                            tlt=", ".join(tlterm)
                            record.append(slt)
                            record.append(tlt)
                            data.append(record)  
                            record=[]
                        slterm=[]
                        tlterm=[]
                elif event=="start" and tag=="langSec":
                    if elem.attrib["{http://www.w3.org/XML/1998/namespace}lang"]==sl:
                        lang=sl
                    if elem.attrib["{http://www.w3.org/XML/1998/namespace}lang"]==tl:
                        lang=tl
                elif event=="start" and tag=="term":
                    if lang==sl: slterm.append("".join(elem.itertext()).lstrip().rstrip())
                    elif lang==tl: tlterm.append("".join(elem.itertext()).lstrip().rstrip())
            
            self.cur.executemany("INSERT INTO reference_terms (sl_term,tl_term) VALUES (?,?)",data)   
            self.conn.commit()
                    
        '''
        #<conceptEntry id="37414"><descrip type="subjectField">illness</descrip><langSec xml:lang="en"><termSec><term>azaprocin</term><termNote type="termType">fullForm</termNote><descrip type="reliabilityCode">9</descrip></termSec></langSec><langSec xml:lang="es"><termSec><term>azaprocina</term><termNote type="termType">fullForm</termNote><descrip type="reliabilityCode">9</descrip></termSec></langSec></conceptEntry><conceptEntry id="1892703"><descrip type="subjectField">chemistry;pharmaceutical industry</descrip><langSec xml:lang="en"><termSec><term>aclatonium napadisilate</term><termNote type="termType">fullForm</termNote><descrip type="reliabilityCode">9</descr
        cf=codecs.open(arxiu,"r",encoding=encoding)
        data=[]
        continserts=0
        for line in cf:
            
            line=line.rstrip()
            continserts+=1            
            record=[]
            line=line.rstrip()
            if len(line.split(" "))>=nmin and len(line.split(" "))<=nmax:
                record.append(line)
                #self.tsr_terms[line]="_"
                data.append(record)
        self.cur.executemany("INSERT INTO tsr_terms (term) VALUES (?)",data)
        self.conn.commit()
        '''
    
    def load_tsr_terms(self,arxiu,encoding="utf-8",nmin=0,nmax=1000):
        '''Loads the evaluation terms from a text file'''
        
        cf=codecs.open(arxiu,"r",encoding=encoding)
        data=[]
        continserts=0
        for line in cf:
            
            line=line.rstrip()
            continserts+=1            
            record=[]
            line=line.rstrip()
            if len(line.split(" "))>=nmin and len(line.split(" "))<=nmax:
                record.append(line)
                #self.tsr_terms[line]="_"
                data.append(record)
        self.cur.executemany("INSERT INTO tsr_terms (term) VALUES (?)",data)
        self.conn.commit()
    
    def load_compoundify_terms_sl(self,arxiu,encoding="utf-8",nmin=0,nmax=1000, field=0):
        '''Loads the evaluation terms from a text file'''
        
        cf=codecs.open(arxiu,"r",encoding=encoding)
        data=[]
        continserts=0
        for line in cf:
            
            line=line.rstrip()
            camps=line.split("\t")
            line=camps[field]
            continserts+=1            
            record=[]
            line=line.rstrip()
            if len(line.split(" "))>=nmin and len(line.split(" "))<=nmax:
                record.append(line)
                #self.tsr_terms[line]="_"
                data.append(record)
        self.cur.executemany("INSERT INTO compoundify_terms_sl (term) VALUES (?)",data)
        self.conn.commit()
        
    def load_compoundify_terms_tl(self,arxiu,encoding="utf-8",nmin=0,nmax=1000, field=0):
        '''Loads the evaluation terms from a text file'''
        
        cf=codecs.open(arxiu,"r",encoding=encoding)
        data=[]
        continserts=0
        for line in cf:
            line=line.rstrip()
            camps=line.split("\t")
            line=camps[field]
            continserts+=1            
            record=[]
            line=line.rstrip()
            if len(line.split(" "))>=nmin and len(line.split(" "))<=nmax:
                record.append(line)
                #self.tsr_terms[line]="_"
                data.append(record)
        self.cur.executemany("INSERT INTO compoundify_terms_tl (term) VALUES (?)",data)
        self.conn.commit()
    
    def load_exclusion_terms(self,arxiu,encoding="utf-8",nmin=0,nmax=1000):
        '''Loads the exclusion terms from a tabulated text. The terms in the exclusion terms will be deleted from the term candidates. It is useful to store already known terms, and/or term candidates already evaluated either as correct or as incorrect.'''
        self.cf=codecs.open(arxiu,"r",encoding=encoding)
        self.data=[]
        self.continserts=0
        for self.line in self.cf:
            self.line=self.line.rstrip()
            self.continserts+=1            
            self.record=[]
            self.line=self.line.rstrip()
            self.camps=self.line.split("\t")
            if len(self.camps)==1:
                self.record.append(self.camps[0])
                self.record.append("_")
                self.evaluation_terms[self.camps[0]]="_"
            elif len(self.camps)>1:
                self.record.append(self.camps[0])
                self.record.append(self.camps[1])
                self.evaluation_terms[self.camps[0]]=self.camps[1]
            self.data.append(self.record)
            if self.continserts==self.maxinserts:
                self.cur.executemany("INSERT INTO exclusion_terms (sl_term,tl_term) VALUES (?,?)",self.data)
                self.data=[]
                self.continserts=0
        with self.conn:
            self.cur.executemany("INSERT INTO exclusion_terms (sl_term,tl_term) VALUES (?,?)",self.data)
    
        self.conn.commit()

    def load_sl_exclusion_regexps(self,arxiu,encoding="utf-8"):
        '''Loads the exclusionregular expressions for the source language.'''
        self.cf=codecs.open(arxiu,"r",encoding=encoding)
        self.data=[]
        for self.line in self.cf:
            self.line=self.line.rstrip()
            self.record=[]
            self.record.append(self.line)
            self.data.append(self.record)
            
        with self.conn:
            self.cur.executemany('INSERT INTO sl_exclusion_regexps (sl_exclusion_regexp) VALUES (?)',self.data)
            
    def load_tl_exclusion_regexps(self,arxiu,encoding="utf-8"):
        '''Loads the exclusionregular expressions for the target language.'''
        self.cf=codecs.open(arxiu,"r",encoding=encoding)
        self.data=[]
        for self.line in self.cf:
            self.line=self.line.rstrip()
            self.record=[]
            self.record.append(self.line)
            self.data.append(self.record)
            
        with self.conn:
            self.cur.executemany('INSERT INTO tl_exclusion_regexps (sl_exclusion_regexp) VALUES (?)',self.data)

   
    def show_term_candidates(self,limit=-1,minfreq=2, minmeasure=-1, show_frequency=True, show_measure=False, mark_eval=False, verbose=False):
        '''Shows in screen the term candidates.'''
        self.measure=0
        
        with self.conn:
            self.cur.execute("SELECT sl_term FROM exclusion_terms")
            for self.s in self.cur.fetchall():
                self.knownterms.append(self.s[0])
        with self.conn:
            self.cur.execute("SELECT frequency,value,n,candidate FROM term_candidates order by value desc, frequency desc, random() limit "+str(limit))
            for self.s in self.cur.fetchall():
                self.frequency=self.s[0]
                if self.s[1]==None:
                    self.measure==0
                else:
                    self.measure=self.s[1]
                self.n=self.s[2]
                self.candidate=self.s[3]
                if self.n>=self.n_min and self.n<=self.n_max and not self.candidate in self.knownterms:
                    if mark_eval:
                        if self.candidate in self.evaluation_terms:
                            self.candidate="*"+self.candidate
                    if show_frequency and not show_measure:
                        self.cadena=str(self.frequency)+"\t"+self.candidate
                    if not show_frequency and show_measure:
                        self.cadena=str(self.measure)+"\t"+self.candidate
                    if show_measure and show_frequency:
                        self.cadena=str(self.frequency)+"\t"+str(self.measure)+"\t"+self.candidate
                    else:
                        self.cadena=self.candidate
                    print(self.cadena)

    def select_unigrams(self,file,position=-1,verbose=True):
        sunigrams=codecs.open(file,"w",encoding="utf-8")
        unigrams={}
        self.cur.execute("SELECT frequency,candidate FROM term_candidates order by value desc, frequency desc, random()")
        #self.cur.execute("SELECT frequency,value,n,candidate FROM term_candidates order by n desc limit "+str(limit))
        for s in self.cur.fetchall():
            frequency=s[0]
            candidate=s[1].split(" ")[position]
            if candidate in unigrams:
                unigrams[candidate]+=frequency
            else:
                unigrams[candidate]=frequency
        #for self.candidate in self.unigrams:
        #    print(self.unigrams[self.candidate],self.candidate)
        data=[]
        for candidate in sorted(unigrams, key=unigrams.get, reverse=True):
            
            cadena=str(unigrams[candidate])+"\t"+candidate
            #if self.verbose: print(cadena)
            record=[]
            record.append(candidate)
            record.append(1)
            record.append(unigrams[candidate])
            record.append("freq")
            record.append(unigrams[candidate])
            data.append(record)
            sunigrams.write(cadena+"\n")
          
        with self.conn:
            self.cur.executemany("INSERT INTO term_candidates (candidate, n, frequency, measure, value) VALUES (?,?,?,?,?)",data)        
            self.conn.commit()


    def save_term_candidates(self,outfile,limit=-1,minfreq=2, minmeasure=-1, show_frequency=True, show_measure=False, mark_eval=False, verbose=False):
        '''Saves the term candidates in a file.'''
        self.sortida=codecs.open(outfile,"w",encoding="utf-8")
        self.measure=0
        self.knownterms=[]
        with self.conn:
            self.cur.execute("SELECT sl_term FROM exclusion_terms")
            for self.s in self.cur.fetchall():
                self.knownterms.append(self.s[0])
        with self.conn:
            self.cur.execute("SELECT frequency,value,n,candidate FROM term_candidates order by value desc, frequency desc, random() limit "+str(limit))
            #self.cur.execute("SELECT frequency,value,n,candidate FROM term_candidates order by n desc limit "+str(limit))
            for self.s in self.cur.fetchall():
                self.frequency=self.s[0]
                if self.s[1]==None:
                    self.measure==0
                else:
                    self.measure=self.s[1]
                self.n=self.s[2]
                self.candidate=self.s[3]
                if self.n>=self.n_min and self.n<=self.n_max and not self.candidate in self.knownterms:
                    if mark_eval:
                        if self.candidate in self.evaluation_terms:
                            self.candidate="*"+self.candidate
                    if show_measure and not show_frequency:
                        self.cadena=str(self.measure)+"\t"+self.candidate
                    elif show_frequency and not show_measure:
                        self.cadena=str(self.frequency)+"\t"+self.candidate
                    elif show_frequency and show_measure:
                        self.cadena=str(self.frequency)+"\t"+str(self.measure)+"\t"+self.candidate
                    else:
                        self.cadena=self.candidate
                    if verbose:
                        print(self.cadena)
                    self.sortida.write(self.cadena+"\n")
                    
    #STATISTICAL TERM EXTRACTION
    
    def ngram_calculation (self,nmin,nmax,minfreq=2):
        '''Performs the calculation of ngrams.'''
        ngramsFD=FreqDist()
        tokensFD=FreqDist()
        n_nmin=nmin
        n_max=nmax
            
        with self.conn:
            self.cur.execute('SELECT segment from sl_corpus')
            for s in self.cur.fetchall():
                segment=s[0]
                for n in range(nmin,nmax+1): #we DON'T calculate one order bigger in order to detect nested candidates
                    if self.specificSLtokenizer:
                        tokens=self.SLtokenizer.tokenize(segment).split()
                    else:
                        tokens=segment.split()
                    ngs=ngrams(tokens, n)
                    for ng in ngs:
                        ngramsFD[ng]+=1
                for token in tokens:
                    tokensFD[token]+=1
                       
        data=[]                
        for c in ngramsFD.most_common():
            if c[1]>=minfreq:
                record=[]
                record.append(" ".join(c[0]))            
                record.append(len(c[0]))
                record.append(c[1])   
                data.append(record)
        with self.conn:
            self.cur.executemany("INSERT INTO ngrams (ngram, n, frequency) VALUES (?,?,?)",data) 
            self.conn.commit()
            
        data=[]                
        for c in tokensFD.most_common():
            record=[]
            record.append(c[0])            
            record.append(c[1])   
            data.append(record)
        with self.conn:
            self.cur.executemany("INSERT INTO tokens (token, frequency) VALUES (?,?)",data) 
            self.conn.commit()
            
    def statistical_term_extraction(self,minfreq=2):
        '''Performs an statistical term extraction using the extracted ngrams (ngram_calculation should be executed first) Loading stop-words is advisable. '''
        self.cur.execute("DELETE FROM term_candidates")
        self.conn.commit()
        self.cur.execute("SELECT ngram, n, frequency FROM ngrams order by frequency desc")
        self.results=self.cur.fetchall()
        self.data=[] 
        for self.a in self.results:
            self.ng=self.a[0].split(" ")
            self.include=True
            if self.ng[0].lower() in self.sl_stopwords: self.include=False
            if self.ng[-1].lower() in self.sl_stopwords: self.include=False
            for self.i in range(1,len(self.ng)):
                if self.ng[self.i].lower() in self.sl_inner_stopwords:
                    self.include=False
            if self.include:
                self.record=[]
                self.record.append(self.a[0])            
                self.record.append(self.a[1])
                self.record.append(self.a[2])
                self.record.append("freq")
                self.record.append(self.a[2])   
                self.data.append(self.record)
            if self.a[2]<minfreq:
                break
        with self.conn:
            self.cur.executemany("INSERT INTO term_candidates (candidate, n, frequency, measure, value) VALUES (?,?,?,?,?)",self.data)        
            self.conn.commit()
    def loadSLTokenizer(self, tokenizer):
        if not tokenizer.endswith(".py"): tokenizer=tokenizer+".py"
        spec = importlib.util.spec_from_file_location('', tokenizer)
        tokenizermod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tokenizermod)
        self.SLtokenizer=tokenizermod.Tokenizer()
        self.specificSLtokenizer=True
        
    def unloadSLTokenizer(self):
        self.SLtokenizer=None
        self.specificSLtokenizer=False
        
    def loadTLTokenizer(self, tokenizer):
        if not tokenizer.endswith(".py"): tokenizer=tokenizer+".py"
        spec = importlib.util.spec_from_file_location('', tokenizer)
        tokenizermod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tokenizermod)
        self.TLtokenizer=tokenizermod.Tokenizer()
        self.specificTLtokenizer=True
        
    def unloadSLTokenizer(self):
        self.TLtokenizer=None
        self.specificTLtokenizer=False
            
    def statistical_term_extraction_by_segment(self, segment, minlocalfreq=1, minglobalfreq=2, maxcandidates=2, nmin=1, nmax=4):
        '''Performs an statistical term extraction over a single segment using the extracted ngrams (ngram_calculation should be executed first) Loading stop-words is advisable. '''
        ngramsFD=FreqDist()
        for n in range(nmin,nmax+1): #we DON'T calculate one order bigger in order to detect nested candidates
            if self.specificSLtokenizer:
                tokens=self.SLtokenizer.tokenize(segment).split(" ")
            else:
                tokens=segment.split()
            ngs=ngrams(tokens, n)
            for ng in ngs:
                include=True
                print(ng)
                if ng[0].lower() in self.sl_stopwords: include=False
                if ng[-1].lower() in self.sl_stopwords: include=False
                for i in range(1,len(ng)):
                    if ng[i].lower() in self.sl_inner_stopwords:
                        include=False
                if include: ngramsFD[" ".join(ng)]+=1
                
        for ng in ngramsFD.most_common():
            print(ng)

    def case_normalization(self,verbose=False):
        '''
        Performs case normalization. If a capitalized term exists as non-capitalized, the capitalized one will be deleted and the frequency of the non-capitalized one will be increased by the frequency of the capitalized.
        '''
        self.cur.execute("SELECT candidate,frequency FROM term_candidates order by frequency desc")
        self.results=self.cur.fetchall()
        self.auxiliar={}
        for self.r in self.results:
            self.auxiliar[self.r[0]]=self.r[1]
        for self.a in self.results:
            if not self.a[0]==self.a[0].lower() and self.a[0].lower() in self.auxiliar:
                #self.term_candidates.inc(c[0].lower(),self.term_candidates[c[0]])
                #self.term_candidates[c[0].lower()]+=self.term_candidates[c[0]]
                #self.term_candidates.pop(c[0])     
                self.terma=self.a[0]
                self.termb=self.a[0].lower()
                self.freqa=self.a[1]
                self.freqb=self.auxiliar[self.termb]
                
                self.n=len(self.termb.split(" "))
                self.freqtotal=self.freqa+self.freqb
                if verbose:
                    print(self.terma,self.freqa,"-->",self.termb,self.freqb,"-->",self.freqtotal)
                self.cur.execute('DELETE FROM term_candidates WHERE candidate=?', (self.terma,))
                self.cur.execute('DELETE FROM term_candidates WHERE candidate=?', (self.termb,))
                #self.cur.execute("INSERT INTO term_candidates (candidate, n, frequency) VALUES (?,?,?)",(self.termb,self.n,self.freqtotal,))
                self.cur.execute("INSERT INTO term_candidates (candidate, n, frequency, measure, value) VALUES (?,?,?,?,?)",(self.termb,self.n,self.freqtotal,"freq",self.freqtotal))
        self.conn.commit()

    def nest_normalization(self,percent=10,verbose=False):
        '''
        Performs a normalization of nested term candidates. If an n-gram candidate A is contained in a n+1 candidate B and freq(A)==freq(B) or they are close values (determined by the percent parameter, A is deleted B remains as it is)
        '''
        self.cur.execute("SELECT candidate,frequency,n FROM term_candidates order by frequency desc")
        self.results=self.cur.fetchall()
        for self.a in self.results:
            
            self.ta=self.a[0]
            self.fa=self.a[1]
            self.na=self.a[2]
            self.nb=self.na+1
            self.fmax=self.fa+self.fa*percent/100
            self.fmin=self.fa-self.fa*percent/100
            self.cur.execute("SELECT candidate,frequency FROM term_candidates where frequency <="+str(self.fmax)+" and frequency>="+str(self.fmin)+"  and n ="+str(self.nb)+ " and n<="+str(self.n_max))
            self.results2=self.cur.fetchall()
            for self.b in self.results2:
                self.tb=self.b[0]
                self.fb=self.b[1]
                #if abs(self.fa-self.fb)<=max(self.fa,self.fb)/percent:
                if not self.ta==self.tb and not self.tb.find(self.ta)==-1: 
                
                    self.cur.execute('DELETE FROM term_candidates WHERE candidate=?', (self.ta,))
                    if verbose:
                        print(str(self.fa),self.ta,"-->",str(self.fb),self.tb)

        self.conn.commit()

    def regexp_exclusion(self,verbose=False):
        '''Deletes term candidates matching a set of regular expresions loaded with the load_sl_exclusion_regexps method.'''
        self.cur.execute("SELECT sl_exclusion_regexp FROM sl_exclusion_regexps")
        self.results=self.cur.fetchall()
        for self.r in self.results:
            self.nregexp=len(self.r[0].split(" "))
            self.exreg=self.r[0]
            self.cur.execute("SELECT candidate FROM term_candidates where n<="+str(self.n_max))
            self.results=self.cur.fetchall()
            self.cexreg=re.compile(self.exreg)
            for self.a in self.results:
                self.candidate=self.a[0]
                self.ncandidate=len(self.candidate.split(" "))
                self.match=re.match(self.cexreg,self.candidate)
                #self.match=re.match(r'\W+$',self.candidate)
                
                if not self.match==None and self.nregexp==self.ncandidate:
                    self.cur.execute('DELETE FROM term_candidates WHERE candidate=?', (self.candidate,))
                    if verbose:
                        print(self.exreg,"-->",self.candidate)
            self.conn.commit()

    #EVALUATION
    
    
     
    def evaluate_pos(self,limit,order="desc",iterations=1000,ignore_case=True):
        '''Performs the evaluation of the term candidates using the evaluation_terms loaded with the load_evaluation_terms method.'''
        self.correct=0
        self.total=0
        self.evaluation_terms=[]
        self.cur.execute("SELECT sl_term FROM evaluation_terms")
        results=self.cur.fetchall()
        for r in results:
            self.evaluation_terms.append(r[0])
        self.tsr_terms=[]
        self.cur.execute("SELECT term FROM tsr_terms")
        results=self.cur.fetchall()
        for r in results:
            self.tsr_terms.append(r[0])
        self.evaluation_terms.extend(self.tsr_terms)
        with self.conn:
            for i in range(0,iterations):
                if order=="desc":
                    self.cur.execute("SELECT candidate,value from term_candidates where n<="+str(self.n_max)+" order by value desc, frequency desc, random() limit "+str(limit))
                elif order=="asc":
                    self.cur.execute("SELECT candidate from term_candidates where n<="+str(self.n_max)+" order by value asc, frequency desc, random() limit "+str(limit))
                else:
                    raise NameError('Order must be desc (decending) or asc (ascending). Defaulf value: desc')
                #self.cur.execute("SELECT candidate from term_candidates order by id limit "+str(limit))
                for self.s in self.cur.fetchall():
                    self.total+=1
                    self.candidate=self.s[0]
                    if ignore_case:
                        if self.candidate in self.evaluation_terms:
                            self.correct+=1
                        elif self.candidate.lower() in self.evaluation_terms:
                            self.correct+=1
                    else:
                        if self.candidate in self.evaluation_terms:
                            self.correct+=1
            self.correct=self.correct/iterations
            self.total=self.total/iterations
            
        try:
            self.precisio=100*self.correct/self.total
            self.recall=100*self.correct/len(self.evaluation_terms)
            self.f1=2*self.precisio*self.recall/(self.precisio+self.recall)
            return(limit,self.correct,self.total,self.precisio,self.recall,self.f1)
        except:
            return(limit,0,0,0,0,0)

    def association_measures(self,measure="raw_freq"):
        measurename=measure
        bigram_measures = myBigramAssocMeasures()
        trigram_measures = myTrigramAssocMeasures()
        quadgram_measures = myQuadgramAssocMeasures()
        
        fd_tokens=nltk.FreqDist()
        fd_bigrams=nltk.FreqDist()
        fd_trigrams=nltk.FreqDist()
        fd_quadgrams=nltk.FreqDist()
        wildcard_fd=nltk.FreqDist()
        self.cur.execute("SELECT token,frequency from tokens")
        for s in self.cur.fetchall():
            aux=(s[0])
            fd_tokens[aux]+=s[1]
            
        textcorpus=[]
        self.cur.execute("SELECT segment from sl_corpus")
        for segment in self.cur.fetchall():
            textcorpus.extend(segment[0].split(" "))
            
        bigram_finder=BigramCollocationFinder.from_words(textcorpus)
        trigram_finder=TrigramCollocationFinder.from_words(textcorpus)
        quadgram_finder=QuadgramCollocationFinder.from_words(textcorpus)
        
        self.cur.execute("SELECT ngram,frequency,n from ngrams")
        results=self.cur.fetchall()
        for r in results:
            data=[]
            data.append(r[0])
            self.cur2.execute("UPDATE term_candidates SET value=NULL where candidate=?",data)
        self.conn.commit()
        data=[]
        bigram_measure=[]
        try:
            bigram_measure=eval("bigram_finder.score_ngrams(bigram_measures."+measure+")")
        except:
            print("ERROR: measure "+measure+ " not implemented for bigrams",sys.exc_info())
            #sys.exit()
            
        for nose in bigram_measure:
            record=[]
            term_candidate=" ".join(nose[0])
            mvalue=nose[1]
            record.append(measure)
            record.append(mvalue)
            record.append(term_candidate)
            data.append(record)
        
        trigram_measure=[]        
        try:
            trigram_measure=eval("trigram_finder.score_ngrams(trigram_measures."+measure+")")
        except:
            print("ERROR: measure "+measure+ " not implemented for trigrams")
            #sys.exit()
        for nose in trigram_measure:
            record=[]
            term_candidate=" ".join(nose[0])
            mvalue=nose[1]
            record.append(measure)
            record.append(mvalue)
            record.append(term_candidate)
            data.append(record)
        quadgram_measure=[]  
        try:
            quadgram_measure=eval("quadgram_finder.score_ngrams(quadgram_measures."+measure+")")
        except:
            print("ERROR: measure "+measure+ " not implemented for quadgrams")
            #sys.exit()
        
        for nose in quadgram_measure:
            record=[]
            term_candidate=" ".join(nose[0])
            mvalue=nose[1]
            record.append(measure)
            record.append(mvalue)
            record.append(term_candidate)
            data.append(record)
            
        self.conn.executemany("UPDATE term_candidates SET measure=?,value=? where candidate=?",data)
        self.conn.commit()
    


    def index_phrase_table(self,phrasetable):
        '''Indexes a phrase table from Moses.'''
        self.entrada=gzip.open(phrasetable, mode='rt',encoding='utf-8')

        self.pt={}
        self.continserts=0
        self.record=[]
        self.data=[]
        while 1:
            self.linia=self.entrada.readline()
            if not self.linia:
                break
            self.linia=self.linia.rstrip()
            self.camps=self.linia.split(" ||| ")
            self.source=self.camps[0].strip()
            self.trad=self.camps[1].strip()
            self.probs=self.camps[2].split(" ")
            try:
                if not self.trad[0] in self.punctuation and not self.source[0] in self.punctuation and not self.trad[-1] in self.punctuation and not self.source[-1] in self.punctuation:
                    #Currently, four different phrase translation scores are computed:
                    #0    inverse phrase translation probability φ(f|e)
                    #1    inverse lexical weighting lex(f|e)
                    #2    direct phrase translation probability φ(e|f)
                    #3    direct lexical weighting lex(e|f)
                    #self.probtrad=float(self.probs[1])
                    self.probtrad=(float(self.probs[2])*float(self.probs[3]))
                    #print(self.source,self.trad,self.probtrad)
                    self.record=[]
                    self.record.append(self.source)
                    self.record.append(self.trad)
                    self.record.append(self.probtrad)
                    self.data.append(self.record)
                    self.continserts+=1
                    if self.continserts==self.maxinserts:
                        self.cur.executemany("INSERT INTO index_pt (source, target, probability) VALUES (?,?,?)",self.data)
                        self.data=[]
                        self.continserts=0
                        self.conn.commit()
            except:
                pass
        with self.conn:
            self.cur.executemany("INSERT INTO index_pt (source, target, probability) VALUES (?,?,?)",self.data)    
        self.conn.commit()
                
        
    def find_translation_ptable(self,sourceterm,maxdec=1,maxinc=1,ncandidates=5,separator=":"):
        '''Finds translation equivalents in an indexed phrase table table. Requires an indexed phrase table and a a list of terms separated by ":".
        The number of translation candidates can be fixed, as well as the maximum decrement and increment of the number of tokens of the translation candidate'''
        #select target from index_pt where source="international conflict";
        self.cur.execute('SELECT target,probability FROM index_pt where source =?',(sourceterm,))
        self.results=self.cur.fetchall()
        self.targetcandidates={}
        for self.a in self.results:
            self.targetterm=self.a[0]
            self.probability=float(self.a[1])
            self.tttokens=self.targetterm.split(" ")
            
            if not self.tttokens[0] in self.tl_stopwords and not self.tttokens[-1] in self.tl_stopwords and len(self.tttokens)>=len(sourceterm.split(" "))-maxdec and len(self.tttokens)<=len(sourceterm.split(" "))+maxinc:
                self.targetcandidates[self.targetterm]=self.probability
        self.sorted_x = sorted(self.targetcandidates.items(), key=operator.itemgetter(1),reverse=True)
        self.results=[]
        for self.s in self.sorted_x:
            self.results.append(self.s[0].replace(":",";"))
        return(separator.join(self.results[0:ncandidates]))
        
                
   
    def start_freeling_api(self,freelingpath, LANG):
        
        if not freelingpath.endswith("/"):freelingpath=freelingpath+"/"
        try:
            sys.path.append(freelingpath+"APIs/python3/")
            import pyfreeling
        except:
            #pass
            print("No Freeling API available. Verify Freeling PATH: "+freelingpath+"freeling/APIs/python3/")
        
        pyfreeling.util_init_locale("default");

        # create language analyzer
        self.la1=pyfreeling.lang_ident(freelingpath+"common/lang_ident/ident.dat");

        # create options set for maco analyzer. Default values are Ok, except for data files.
        self.op1= pyfreeling.maco_options(LANG);
        self.op1.set_data_files( "", 
                           freelingpath + "common/punct.dat",
                           freelingpath+ LANG + "/dicc.src",
                           freelingpath + LANG + "/afixos.dat",
                           "",
                           freelingpath + LANG + "/locucions.dat", 
                           freelingpath + LANG + "/np.dat",
                           freelingpath + LANG + "/quantities.dat",
                           freelingpath + LANG + "/probabilitats.dat");

        # create analyzers
        self.tk1=pyfreeling.tokenizer(freelingpath+LANG+"/tokenizer.dat");
        self.sp1=pyfreeling.splitter(freelingpath+LANG+"/splitter.dat");
        self.sid1=self.sp1.open_session();
        self.mf1=pyfreeling.maco(self.op1);

        # activate mmorpho odules to be used in next call
        #(self, umap: "bool", num: "bool", pun: "bool", dat: "bool",
        # dic: "bool", aff: "bool", comp: "bool", rtk: "bool", 
        # mw: "bool", ner: "bool", qt: "bool", prb: "bool")
        #deactivate mw
        self.mf1.set_active_options(False, True, True, False,  # select which among created 
                              True, True, False, True,  # submodules are to be used. 
                              False, False, True, True ); # default: all created submodules are used

        # create tagger, sense anotator, and parsers
        self.tg1=pyfreeling.hmm_tagger(freelingpath+LANG+"/tagger.dat",True,2);
        
    def tag_freeling_api(self,corpus="source"):
        with self.conn:
            self.data=[]
            if corpus=="source":
                self.cur.execute('SELECT id,segment from sl_corpus')
            elif corpus=="target":
                self.cur.execute('SELECT id,segment from tl_corpus')
            for self.s in self.cur.fetchall():
                self.id=self.s[0]
                self.segment=self.s[1]
                self.l1 = self.tk1.tokenize(self.segment);
                self.ls1 = self.sp1.split(self.sid1,self.l1,True);
                self.ls1 = self.mf1.analyze(self.ls1);
                self.ls1 = self.tg1.analyze(self.ls1);
                self.ttsentence=[]
                for self.s in self.ls1 :
                  self.ws = self.s.get_words();
                  for self.w in self.ws :
                    self.form=self.w.get_form()
                    self.lemma=self.w.get_lemma()
                    self.tag=self.w.get_tag()
                    self.ttsentence.append(self.form+"|"+self.lemma+"|"+self.tag)
                self.ttsentence=" ".join(self.ttsentence)
                self.record=[]
                self.record.append(self.id)
                self.record.append(self.ttsentence)
                self.data.append(self.record)
                if self.continserts==self.maxinserts:
                    if corpus=="source":
                        self.cur.executemany("INSERT INTO sl_tagged_corpus (id, tagged_segment) VALUES (?,?)",self.data)
                    if corpus=="target":
                        self.cur.executemany("INSERT INTO tl_tagged_corpus (id, tagged_segment) VALUES (?,?)",self.data)
                    self.data=[]
                    self.continserts=0
            with self.conn:
                if corpus=="source":
                    self.cur.executemany("INSERT INTO sl_tagged_corpus (id, tagged_segment) VALUES (?,?)",self.data) 
                if corpus=="target":
                    self.cur.executemany("INSERT INTO tl_tagged_corpus (id, tagged_segment) VALUES (?,?)",self.data)
    def save_sl_tagged_corpus(self,outputfile,encoding="utf-8"):
        self.sortida=codecs.open(outputfile,"w",encoding=encoding)
        self.cur.execute('SELECT tagged_segment from sl_tagged_corpus')
        for self.s in self.cur.fetchall():
            self.tagged_segment=self.s[0]
            self.sortida.write(self.tagged_segment+"\n")

    
    def tagged_ngram_calculation (self,nmin=2,nmax=3,minfreq=2):
        '''Calculates the tagged ngrams.'''
        self.ngrams=FreqDist()
        self.n_nmin=nmin
        self.n_max=nmax
        with self.conn:
            self.data=[]
            self.record=[]
            self.record.append(self.n_min)
            self.data.append(self.record)
            #self.conn.executemany("UPDATE configuration SET n_min=? where id=0",self.data)
            self.data=[]
            self.record=[]
            self.record.append(self.n_max)
            self.data.append(self.record)
            #self.conn.executemany("UPDATE configuration SET n_max=? where id=0",self.data)
            #cur.execute("UPDATE Contacts SET FirstName = ? WHERE LastName = ?", (Fname, Lname))

            self.conn.commit()
            
        with self.conn:
            self.cur.execute('SELECT tagged_segment from sl_tagged_corpus')
            
            for self.s in self.cur.fetchall():
                self.segment=self.s[0]
                
                for self.n in range(nmin,nmax+1):
                    self.ngs=ngrams(self.segment.split(), self.n)
                    for self.ng in self.ngs:
                        self.ngrams[self.ng]+=1
                        
        self.data=[]                
        for self.c in self.ngrams.most_common():
           if self.c[1]>=minfreq:
                self.candidate=[]
                for self.ngt in self.c[0]:
                    self.candidate.append(self.ngt.split("|")[0])
                self.candidate=" ".join(self.candidate)
                self.record=[]
                self.record.append(self.candidate)
                self.record.append(" ".join(self.c[0])) 
                self.record.append(len(self.c[0]))
                self.record.append(self.c[1])   
                self.data.append(self.record)
        with self.conn:
            self.cur.executemany("INSERT INTO tagged_ngrams (ngram, tagged_ngram, n, frequency) VALUES (?,?,?,?)",self.data) 
            self.conn.commit()
                    
    def translate_linguistic_pattern(self,pattern):
           
        self.aux=[]
        for self.ptoken in pattern.split(" "):
            self.auxtoken=[]
            self.ptoken=self.ptoken.replace(".*","[^\s]+") #ATENCIÓ AIXÓ ÉS NOU, VERIFICAR SI CAL
            for self.pelement in self.ptoken.split("|"):
                if self.pelement=="#":
                    self.auxtoken.append("([^\s]+?)")                    
                elif self.pelement=="":
                    self.auxtoken.append("[^\s]+?")
                else:
                    if self.pelement.startswith("#"):
                        self.auxtoken.append("("+self.pelement.replace("#","")+")")
                    else:
                        self.auxtoken.append(self.pelement)
            self.aux.append("\|".join(self.auxtoken))
        self.tp="("+" ".join(self.aux)+")"
        #self.tpatterns.append(self.tp)
        return(self.tp)       
    
    def load_linguistic_patterns(self,file, encoding="utf-8"):
        '''Loads the linguistic patterns to use with linguistic terminology extraction.'''
        self.entrada=codecs.open(file,"r",encoding=encoding)
        self.linguistic_patterns=[]
        self.data=[]
        self.record=[]
        for self.linia in self.entrada:
            self.linia=self.linia.rstrip()
            self.pattern=self.translate_linguistic_pattern(self.linia)
            self.record.append(self.pattern)
            self.data.append(self.record)
            self.record=[]
        with self.conn:
            self.cur.executemany("INSERT INTO linguistic_patterns (linguistic_pattern) VALUES (?)",self.data)
        
         

    def linguistic_term_extraction(self,minfreq=2):
        '''Performs an linguistic term extraction using the extracted tagged ngrams (tagged_ngram_calculation should be executed first). '''
        self.linguistics_patterns=[]
        self.controlpatterns=[]
        with self.conn:
            self.cur.execute("SELECT linguistic_pattern from linguistic_patterns")
            for self.lp in self.cur.fetchall():
                self.linguistic_pattern=self.lp[0]
                self.transformedpattern="^"+self.linguistic_pattern+"$"
                if not self.transformedpattern in self.controlpatterns:
                    self.linguistic_patterns.append(self.transformedpattern)
                    self.controlpatterns.append(self.transformedpattern)
                    
            
        self.cur.execute("SELECT tagged_ngram, n, frequency FROM tagged_ngrams order by frequency desc")
        self.results=self.cur.fetchall()
        self.data=[] 
        
        for self.a in self.results:
            self.include=True
            self.ng=self.a[0]
            self.n=self.a[1]
            self.frequency=self.a[2]
            try:
                if self.ng.split(" ")[0].split("|")[1].lower() in self.sl_stopwords: self.include=False
            except:
                pass
            try:
                if self.ng.split(" ")[-1].split("|")[1].lower() in self.sl_stopwords: self.include=False
            except:
                pass
            if self.frequency<minfreq:
                break
            if self.include:
                for self.pattern in self.linguistic_patterns:
                    self.match=re.search(self.pattern,self.ng)
                    if self.match:
                        if self.match.group(0)==self.ng:          
                            self.candidate=" ".join(self.match.groups()[1:])
                            self.record=[]
                            self.record.append(self.candidate)     
                            self.record.append(self.n)
                            self.record.append(self.frequency)   
                            self.record.append("freq")
                            self.record.append(self.frequency)   
                            self.data.append(self.record)
                            break
                
        with self.conn:
            #self.cur.executemany("INSERT INTO term_candidates (candidate, n, frequency) VALUES (?,?,?)",self.data)  
            self.cur.executemany("INSERT INTO term_candidates (candidate, n, frequency, measure, value) VALUES (?,?,?,?,?)",self.data)      
            self.conn.commit()
            
        #eliminem candidats repetits
        self.cur.execute("SELECT candidate, n, frequency FROM term_candidates")
        self.results=self.cur.fetchall()
        self.tcaux={}
        for self.a in self.results:
            if not self.a[0] in self.tcaux:
                self.tcaux[self.a[0]]=self.a[2]
            else:
                self.tcaux[self.a[0]]+=self.a[2]
        
        self.cur.execute("DELETE FROM term_candidates")
        self.conn.commit()
        self.data=[] 
        for self.tc in self.tcaux:
            self.record=[]
            self.record.append(self.tc)            
            self.record.append(len(self.tc.split(" ")))
            self.record.append(self.tcaux[self.tc])   
            self.record.append("freq")
            self.record.append(self.tcaux[self.tc])   
            self.data.append(self.record)
        with self.conn:
            #self.cur.executemany("INSERT INTO term_candidates (candidate, n, frequency) VALUES (?,?,?)",self.data)        
            self.cur.executemany("INSERT INTO term_candidates (candidate, n, frequency, measure, value) VALUES (?,?,?,?,?)",self.data) 
            self.conn.commit()
            
     
    def learn_linguistic_patterns(self,outputfile,showfrequencies=False,encoding="utf-8",verbose=True,representativity=100):
        self.learntpatterns={}
        self.sortida=codecs.open(outputfile,"w",encoding=encoding)
        self.acufreq=0
        tags={}
        with self.conn:
            self.cur.execute("SELECT sl_term FROM evaluation_terms")
            for self.s in self.cur.fetchall():
                self.cur.execute("SELECT tagged_ngram, n, frequency FROM tagged_ngrams WHERE ngram= ?", (self.s[0],))
                self.results=self.cur.fetchall()
                if len(self.results)>0:
                    for self.a in self.results:
                        self.ng=self.a[0]
                        nglist=self.ng.split()
                        self.n=self.a[1]
                        self.frequency=self.a[2]
                        self.candidate=[]
                        self.ngtokenstag=self.ng.split(" ")
                        for self.ngt in self.ngtokenstag:
                            self.candidate.append(self.ngt.split("|")[0])
                        self.candidate=" ".join(self.candidate)
                        self.t2=self.ng.split(" ")
                        self.t1=self.candidate.split(" ")
                        self.patternbrut=[]
                        for self.position in range(0,self.n):
                            self.t2f=self.t2[self.position].split("|")[0]
                            self.t2l=self.t2[self.position].split("|")[1]
                            self.t2t=self.t2[self.position].split("|")[2]
                            self.patternpart=""
                            if self.t1[self.position]==self.t2l:
                                self.patternpart="|#|"+self.t2t
                            elif self.t1[self.position]==self.t2f:
                                self.patternpart="#||"+self.t2t
                            self.patternbrut.append(self.patternpart)
                        self.pattern=" ".join(self.patternbrut)
                        if self.pattern in self.learntpatterns:
                            self.learntpatterns[self.pattern]+=self.n
                            self.acufreq+=self.n
                        else:
                            self.learntpatterns[self.pattern]=self.n
                            self.acufreq+=self.n
        self.sorted_x = sorted(self.learntpatterns.items(), key=operator.itemgetter(1),reverse=True)
        self.results=[]
        self.acufreq2=0
        for self.s in self.sorted_x:
            self.percent=100*self.acufreq2/self.acufreq
            if self.percent>representativity:
                break
            self.acufreq2+=self.s[1]
            if showfrequencies:
                cadena=str(self.s[1])+"\t"+self.s[0]
            else:
                cadena=self.s[0]
            self.sortida.write(cadena+"\n")
            if verbose:
                print(cadena)
                                
    def find_translation_pcorpus(self,slterm,maxdec=1,maxinc=1,ncandidates=5,separator=":"):
        self.nmin=len(slterm.split(" "))-maxdec
        self.nmax=len(slterm.split(" "))+maxinc
        self.tlngrams=FreqDist()
        with self.conn:
            self.cur.execute('SELECT id, segment from sl_corpus')
            
            for self.s in self.cur.fetchall():
                self.segment=self.s[1]
                self.id=self.s[0]
                
                if self.segment.find(slterm)>-1:
                    self.cur2.execute('SELECT segment from tl_corpus where id="'+str(self.id)+'"')
                    for self.s2 in self.cur2.fetchall():
                        self.tl_segment=self.s2[0]
                        for self.n in range(self.nmin,self.nmax+1):
                            #self.tlngs=ngrams(self.tl_tokenizer.tokenize(self.tl_segment), self.n)
                            self.tlngs=ngrams(self.tl_segment.split(" "), self.n)
                            for self.tlng in self.tlngs:
                                if not self.tlng[0] in self.tl_stopwords and not self.tlng[-1] in self.tl_stopwords:
                                    self.tlngrams[self.tlng]+=1
            self.resultlist=[]
            for self.c in self.tlngrams.most_common(ncandidates):
                self.resultlist.append(" ".join(self.c[0]))
            
            return(separator.join(self.resultlist))

#EMBEDDINGS

    def calculate_embeddings_sl(self,filename,vector_size=300, window=5, min_count=1, workers=4):
        self.cur.execute('SELECT id, segment from sl_corpus')
        data = []
        for s in self.cur.fetchall():
            temp=[]
            segment=s[1] 
            if self.specificSLtokenizer:
                tokens=self.SLtokenizer.tokenize(segment).split(" ")
            else:
                tokens=segment.split()
            data.append(tokens)
        model = Word2Vec(sentences=data, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        model.wv.save_word2vec_format(filename, binary=False)
     
    def calculate_embeddings_sl_ref(self,filename,vector_size=300, window=5, min_count=1, workers=4):
        self.cur.execute('SELECT id, segment from tl_corpus')
        data = []
        for s in self.cur.fetchall():
            temp=[]
            segment=s[1] 
            if self.specificSLtokenizer:
                tokens=self.SLtokenizer.tokenize(segment).split(" ")                
            else:
                tokens=segment.split()
            data.append(tokens)
        model = Word2Vec(sentences=data, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        model.wv.save_word2vec_format(filename, binary=False)
    
    def calculate_embeddings_tl(self,filename,vector_size=300, window=5, min_count=1, workers=4):
        self.cur.execute('SELECT id, segment from tl_corpus')
        data = []
        for s in self.cur.fetchall():
            temp=[]
            segment=s[1] 
            if self.specificTLtokenizer:
                tokens=self.TLtokenizer.tokenize(segment).split(" ")
            else:
                tokens=segment.split()
            data.append(tokens)
        model = Word2Vec(sentences=data, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        model.wv.save_word2vec_format(filename, binary=False)
    
    def mapEmbeddings(self,src_input,trg_input,src_output,trg_output,init_dictionary):
        map_embeddings.supervised_mapping(src_input,trg_input,src_output,trg_output,init_dictionary)
        
    def load_SL_embeddings(self, file, binary=False):
        self.wvSL = KeyedVectors.load_word2vec_format(file, binary=False)
        
    def load_TL_embeddings(self, file, binary=False):
        self.wvTL = KeyedVectors.load_word2vec_format(file, binary=False)
        
    
    def find_translation_wv(self, term, ncandidates=10):
        term=term.strip().replace(" ","▁")
        try:
            vector=self.wvSL[term]
            tcandidates = self.wvTL.most_similar([vector], topn=ncandidates)
        except:
            tcandidates=[]
        resposta=[]
        for tc in tcandidates:
            tc=tc[0].replace("▁"," ")
            resposta.append(tc)
        return(resposta)
        


#TSR
    def tsr(self, type="combined",max_iterations=10000000000, verbose=True): 
        component={}
        firstcomponent={}
        middlecomponent={}
        lastcomponent={}
        self.tsr_terms=[]
        self.cur.execute("SELECT term FROM tsr_terms")
        results=self.cur.fetchall()
        for r in results:
            self.tsr_terms.append(r[0])
        for term in self.tsr_terms:
            camps=term.split(" ")
            if len(camps)==1: #UNIGRAMS
                firstcomponent[camps[0].lower()]=1
                lastcomponent[camps[0].lower()]=1
            if len(camps)>=2:
                firstcomponent[camps[0].lower()]=1
                lastcomponent[camps[-1].lower()]=1
                component[camps[0].lower()]=1
                component[camps[-1].lower()]=1
                if len(camps)>=3:
                    for i in range(1,len(camps)-1):
                        middlecomponent[camps[i].lower()]=1
                        component[camps[i].lower()]=1

        new=True
        newcandidates={} #candidate-frequency
        hashmeasure={}
        hashvalue={}
        
        newcandidatestempstric={} #candidate-frequency                                                                    
        hashmeasuretempstrict={}
        hashvaluetempstric={}

        newcandidatestempflexible={} #candidate-frequency
        hashmeasuretempflexible={}
        hashvaluetempflexible={}
        
        newcandidatestempcombined={} #candidate-frequency
        hashmeasuretempcombined={}
        hashvaluetempcombined={}
        
        iterations=0
        while new:
            iterations+=1
            if verbose: print("ITERATION",iterations)
            new=False
            self.cur.execute("SELECT candidate,n,frequency,measure,value FROM term_candidates ")
            results=self.cur.fetchall()
            auxiliar={}
            value=max_iterations-iterations#r[4]
            for r in results:
                candidate=r[0]
                n=r[1]
                frequency=r[2]
                measure="tsr"#r[3]
                #IMPLEMENTED ONLY FOR BIGRAMS !!!
                '''
                rcamps=candidate.split(" ")
                if type=="strict":
                    if rcamps[0] in firstcomponent and rcamps[-1] in lastcomponent:
                        if not candidate in newcandidates:
                            newcandidates[candidate]=frequency
                            hashmeasure[candidate]=measure
                            hashvalue[candidate]=value
                            new=True
                            firstcomponent[rcamps[0]]=1
                            lastcomponent[rcamps[-1]]=1
                elif type=="flexible":
                    if rcamps[0] in firstcomponent or rcamps[-1] in lastcomponent:
                        if not candidate in newcandidates:
                            newcandidates[candidate]=frequency
                            hashmeasure[candidate]=measure
                            hashvalue[candidate]=value
                            new=True
                            firstcomponent[rcamps[0]]=1
                            lastcomponent[rcamps[-1]]=1
                            component[rcamps[0]]=1
                            component[rcamps[-1]]=1
                elif type=="combined":
                    if iterations==1:
                        if rcamps[0] in firstcomponent and rcamps[-1] in lastcomponent:
                            if not candidate in newcandidates:
                                newcandidates[candidate]=frequency
                                hashmeasure[candidate]=measure
                                hashvalue[candidate]=value
                                new=True
                                firstcomponent[rcamps[0]]=1
                                lastcomponent[rcamps[-1]]=1
                                component[rcamps[0]]=1
                                component[rcamps[-1]]=1
                    else:
                        if rcamps[0] in firstcomponent or rcamps[-1] in lastcomponent:
                            if not candidate in newcandidates:
                                newcandidates[candidate]=frequency
                                hashmeasure[candidate]=measure
                                hashvalue[candidate]=value
                                new=True
                                firstcomponent[rcamps[0]]=1
                                lastcomponent[rcamps[-1]]=1
                                component[rcamps[0]]=1
                                component[rcamps[-1]]=1
                '''
                first_c=False
                middle_c=False
                last_c=False
                rcamps=candidate.split(" ")
                truesfalses=[]
                if str(rcamps[0]).lower() in firstcomponent: 
                    first_c=True
                    truesfalses.append(True)
                else:
                    truesfalses.append(False)
                if str(rcamps[-1]).lower() in lastcomponent: 
                    last_c=True
                    truesfalses.append(True)
                else:
                    truesfalses.append(False)
                if n>2:
                    middle_c=True
                    for i in range(1,n-1):
                        if not str(r[i]).lower() in middlecomponent: middle_c=False
                    if middle_c==True:
                        truesfalses.append(True)
                    else:
                        truesfalses.append(False)
                if type=="strict":
                    if not False in truesfalses:
                        if not candidate in newcandidates:
                            newcandidates[candidate]=frequency
                            hashmeasure[candidate]=measure
                            hashvalue[candidate]=value
                            new=True
                            firstcomponent[rcamps[0]]=1
                            lastcomponent[rcamps[-1]]=1
                elif type=="flexible":
                    if True in truesfalses:
                        if not candidate in newcandidates:
                            newcandidates[candidate]=frequency
                            hashmeasure[candidate]=measure
                            hashvalue[candidate]=value
                            new=True
                            firstcomponent[rcamps[0]]=1
                            lastcomponent[rcamps[-1]]=1
                            component[rcamps[0]]=1
                            component[rcamps[-1]]=1
                elif type=="combined":
                    if iterations==1:       
                        new=True
                        if not False in truesfalses:
                            if not candidate in newcandidates:
                                newcandidates[candidate]=frequency
                                hashmeasure[candidate]=measure
                                hashvalue[candidate]=value                                
                                firstcomponent[rcamps[0]]=1
                                lastcomponent[rcamps[-1]]=1
                                if n>2:
                                    for i in range(1,n-1):
                                        middlecomponent[rcamps[i]]=1
                                        component[rcamps[i]]=1
                    else:
                        if True in truesfalses:
                            if not candidate in newcandidates:
                                newcandidates[candidate]=frequency
                                hashmeasure[candidate]=measure
                                hashvalue[candidate]=value
                                new=True
                                firstcomponent[rcamps[0]]=1
                                lastcomponent[rcamps[-1]]=1
                                if n>2:
                                    for i in range(1,n-1):
                                        middlecomponent[rcamps[i]]=1
                                        component[rcamps[i]]=1
                                component[rcamps[0]]=1
                                component[rcamps[-1]]=1
                '''
                if n==2:
                    if rcamps[0] in firstcomponent: first_c=True
                    middle_c=True
                    if rcamps[-1] in lastcomponent: last_c=True
                    if type=="strict":
                        if first_c and last_c:
                            if not candidate in newcandidates:
                                newcandidates[candidate]=frequency
                                hashmeasure[candidate]=measure
                                hashvalue[candidate]=value
                                new=True
                                firstcomponent[rcamps[0]]=1
                                lastcomponent[rcamps[-1]]=1
                    elif type=="flexible":
                        if first_c or last_c:
                            if not candidate in newcandidates:
                                newcandidates[candidate]=frequency
                                hashmeasure[candidate]=measure
                                hashvalue[candidate]=value
                                new=True
                                firstcomponent[rcamps[0]]=1
                                lastcomponent[rcamps[-1]]=1
                                component[rcamps[0]]=1
                                component[rcamps[-1]]=1
                    
                    elif type=="combined":
                        if iterations==1:
                            if first_c and last_c:
                                if not candidate in newcandidates:
                                    newcandidates[candidate]=frequency
                                    hashmeasure[candidate]=measure
                                    hashvalue[candidate]=value
                                    new=True
                                    firstcomponent[rcamps[0]]=1
                                    lastcomponent[rcamps[-1]]=1
                                    component[rcamps[0]]=1
                                    component[rcamps[-1]]=1
                        else:
                            if first_c or last_c:
                                if not candidate in newcandidates:
                                    newcandidates[candidate]=frequency
                                    hashmeasure[candidate]=measure
                                    hashvalue[candidate]=value
                                    new=True
                                    firstcomponent[rcamps[0]]=1
                                    lastcomponent[rcamps[-1]]=1
                                    component[rcamps[0]]=1
                                    component[rcamps[-1]]=1
                                    
                elif n==3:
                    if rcamps[0] in firstcomponent: first_c=True
                    if rcamps[1] in middlecomponent: middle_c=True
                    if rcamps[-1] in lastcomponent: last_c=True
                    if type=="strict":
                        if first_c and middle_c and last_c:
                            if not candidate in newcandidates:
                                newcandidates[candidate]=frequency
                                hashmeasure[candidate]=measure
                                hashvalue[candidate]=value
                                new=True
                                firstcomponent[rcamps[0]]=1
                                middlecomponent[rcamps[1]]=1
                                lastcomponent[rcamps[-1]]=1
                    elif type=="flexible":
                        condition=False
                        if first_c and middle_c or last_c: condition=True
                        #if first_c or middle_c and last_c: condition=True
                        if last_c and middle_c or first_c: condition=True
                        if condition:
                            if not candidate in newcandidates:
                                newcandidates[candidate]=frequency
                                hashmeasure[candidate]=measure
                                hashvalue[candidate]=value
                                new=True
                                firstcomponent[rcamps[0]]=1
                                middlecomponent[rcamps[1]]=1
                                lastcomponent[rcamps[-1]]=1
                                component[rcamps[0]]=1
                                component[rcamps[1]]=1
                                component[rcamps[-1]]=1
                    
                    elif type=="combined":
                        if iterations==1:
                            if first_c and middle_c and last_c:
                                if not candidate in newcandidates:
                                    newcandidates[candidate]=frequency
                                    hashmeasure[candidate]=measure
                                    hashvalue[candidate]=value
                                    new=True
                                    firstcomponent[rcamps[0]]=1
                                    middlecomponent[rcamps[1]]=1
                                    lastcomponent[rcamps[-1]]=1
                                    component[rcamps[0]]=1
                                    component[rcamps[1]]=1
                                    component[rcamps[-1]]=1
                        else:
                            condition=False
                            if first_c or middle_c or last_c: condition=True
                            #if first_c and middle_c or last_c: condition=True
                            #if first_c or middle_c and last_c: condition=True
                            #if last_c and middle_c or first_c: condition=True
                            if condition:
                                if not candidate in newcandidates:
                                    newcandidates[candidate]=frequency
                                    hashmeasure[candidate]=measure
                                    hashvalue[candidate]=value
                                    new=True
                                    firstcomponent[rcamps[0]]=1
                                    middlecomponent[rcamps[1]]=1
                                    lastcomponent[rcamps[-1]]=1
                                    component[rcamps[0]]=1
                                    component[rcamps[1]]=1
                                    component[rcamps[-1]]=1
            '''     
            if iterations>=max_iterations:
                break
            if verbose: print(iterations,new)
        with self.conn:
            self.cur.execute('DELETE FROM term_candidates')
            self.conn.commit()
        
                    
        data=[]
        for c in newcandidates:
            termb=c
            n=len(c.split(" "))
            freqtotal=newcandidates[c]
            measure=hashmeasure[c]
            value=hashvalue[c]
            record=[]
            record.append(termb)
            record.append(n)
            record.append(freqtotal)
            record.append(measure)
            record.append(value)
            data.append(record)
        with self.conn:
            self.cur.executemany("INSERT INTO term_candidates (candidate, n, frequency, measure, value) VALUES (?,?,?,?,?)",data)
            
        self.conn.commit()   

def L_LLR(a,b,c):
    '''Auxiliar function to calculate Log Likelihood Ratio'''
    L=(c**a)*((1-c)**(b-a))
    return(L)
            
class myBigramAssocMeasures(nltk.collocations.BigramAssocMeasures):
    """
    A collection of bigram association measures. Each association measure
    is provided as a function with three arguments::

        bigram_score_fn(n_ii, (n_ix, n_xi), n_xx)

    The arguments constitute the marginals of a contingency table, counting
    the occurrences of particular events in a corpus. The letter i in the
    suffix refers to the appearance of the word in question, while x indicates
    the appearance of any word. Thus, for example:

        n_ii counts (w1, w2), i.e. the bigram being scored
        n_ix counts (w1, *)
        n_xi counts (*, w2)
        n_xx counts (*, *), i.e. any bigram

    This may be shown with respect to a contingency table::

                w1    ~w1
             ------ ------
         w2 | n_ii | n_oi | = n_xi
             ------ ------
        ~w2 | n_io | n_oo |
             ------ ------
             = n_ix        TOTAL = n_xx
             
    Amb la terminologia de Pazienza

                w1    ~w1
             ------ ------
         w2 | n_ii O11| n_oi O12| = n_xi
             ------ ------
        ~w2 | n_io O21| n_oo O22|
             ------ ------
             = n_ix        TOTAL = n_xx
             
             N=O11+O12+O21+O22= n_xx
             R1=O11+O12 = n_xi
             R2=O21+O22
             C1=O11+O21=n_ix
             C2=O12+O22= n_oi+n_oo
             
        TENIM: n_ii, n_ix_xi_tuple, n_xx
        n_io=n_ix-n_ii
        n_oi=n_xi-n_ii
        n_oo=n_xx-n_ix-n_xi
    """
    #MEASURES NOT IMPLEMENTED IN NLTK
    
    def loglikelihood(self,n_ii, n_ix_xi_tuple, n_xx):
        '''LogLikelihood according to NSP'''
        (n_ix, n_xi) = n_ix_xi_tuple
        n_oo=n_xx-n_ix-n_xi
        n_io=n_ix-n_ii
        n_oi=n_xi-n_ii
        n_oo=n_xx-n_ix-n_xi        
        
        n11=n_ii
        n12=n_io
        n21=n_oi
        n22=n_oo
        n1p=n11+n12
        np1=n11+n21
        n2p=n21+n22
        np2=n12+n22
        npp=n_xx

        m11 = (n1p*np1/npp)
        m12 = (n1p*np2/npp)
        m21 = (np1*n2p/npp)
        m22 = (n2p*np2/npp)
        try:
            LogLikelihood = 2 * (n11 * math.log((n11/m11),2) + n12 * math.log((n12/m12),2) + n21 * math.log((n21/m21),2) + n22 * math.log((n22/m22),2))
        except:
            LogLikelihood=0
        return(LogLikelihood)
        
    def MI(self,n_ii, n_ix_xi_tuple, n_xx):
        '''Church Mutual Information accoding to Pazienza'''
        (n_ix, n_xi) = n_ix_xi_tuple
        self.E11=n_xi*n_ix/n_xx
        self.part=n_ii/self.E11
        self.MI=math.log(self.part,2)
        return(self.MI)
        
    def MI2(self,n_ii, n_ix_xi_tuple, n_xx):
        '''Church Mutual Information Variant accoding to Pazienza'''
        (n_ix, n_xi) = n_ix_xi_tuple
        self.E11=n_xi*n_ix/n_xx
        self.part=(n_ii/self.E11)**2
        self.MI2=math.log(self.part,2)
        return(self.MI2)
        
    def MI3(self,n_ii, n_ix_xi_tuple, n_xx):
        '''Church Mutual Information Variant accoding to Pazienza'''
        (n_ix, n_xi) = n_ix_xi_tuple
        self.E11=n_xi*n_ix/n_xx
        self.part=(n_ii/self.E11)**3
        self.MI3=math.log(self.part,2)
        return(self.MI3)
        
    def odds(self,n_ii, n_ix_xi_tuple, n_xx):
        '''Odds ratio according to NSP'''
        (n_ix, n_xi) = n_ix_xi_tuple
        n_oo=n_xx-n_ix-n_xi
        n_io=n_ix-n_ii
        n_oi=n_xi-n_ii
        n_oo=n_xx-n_ix-n_xi        
        
        n11=n_ii
        n12=n_io
        n21=n_oi
        n22=n_oo
        n1p=n11+n12
        np1=n11+n21
        n2p=n21+n22
        np2=n12+n22
        npp=n_xx

        m11 = (n1p*np1/npp)
        m12 = (n1p*np2/npp)
        m21 = (np1*n2p/npp)
        m22 = (n2p*np2/npp)
        
        if n21==0:n21=1
        if n12==0:n12=1
        ODDS_RATIO = (n11*n22)/(n21*n12)
        return(ODDS_RATIO)
        
    def z_score(self,n_ii, n_ix_xi_tuple, n_xx):
        '''z-score ratio according to NSP'''
        (n_ix, n_xi) = n_ix_xi_tuple
        n_oo=n_xx-n_ix-n_xi
        n_io=n_ix-n_ii
        n_oi=n_xi-n_ii
        n_oo=n_xx-n_ix-n_xi        
        
        n11=n_ii
        n12=n_io
        n21=n_oi
        n22=n_oo
        n1p=n11+n12
        np1=n11+n21
        n2p=n21+n22
        np2=n12+n22
        npp=n_xx

        m11 = (n1p*np1/npp)
        m12 = (n1p*np2/npp)
        m21 = (np1*n2p/npp)
        m22 = (n2p*np2/npp)
       
        zscore = (n11-m11)/(math.sqrt(m11))
        return(zscore)
   

class myTrigramAssocMeasures(nltk.collocations.TrigramAssocMeasures):
    pass
    
class myQuadgramAssocMeasures(nltk.collocations.QuadgramAssocMeasures):
    pass

 
  
    
