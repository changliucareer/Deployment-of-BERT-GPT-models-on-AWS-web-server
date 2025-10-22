
import torch

import time
import random
from uuid import uuid4
from flask import (
    Flask,
    session,
    request,
    redirect,
    url_for,
    render_template_string
)
from planout.experiment import SimpleExperiment
from planout.ops.random import *


class ModelExperiment(SimpleExperiment):
    def setup(self):
        self.set_log_file('model_abtest.log')

    def assign(self, params, userid):
        params.use_pytorch = BernoulliTrial(p=0.5, unit=userid)
        if params.use_pytorch:
            params.model_type = 'BERT'
        else:
            params.model_type = 'GPT-3'



start_time = time.time()


# # Metadata preload
# movies_df = pd.read_csv('movies.dat',
#                         sep="::", header=None, engine='python',
#                         encoding="iso-8859-1")
# movies_df.columns = ['iid', 'name', 'genre']
# movies_df.set_index('iid', inplace=True)
# data = surprise.Dataset.load_builtin('ml-1m')
# trainset = data.build_full_trainset()
# testset = trainset.build_anti_testset()



# Model preload
# k = 100  # latent dimension
# c_bias = 1e-6
# c_vector = 1e-6
# model = MF(trainset.n_users, trainset.n_items,
#            k=k, c_bias=c_bias, c_vector=c_vector)
# model.load_state_dict(torch.load(
#     'recommendation_model_pytorch.pkl'))
# model.eval()
from flask import Flask, jsonify, request, render_template

import time
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
# Load pre-trained model (weights)
model_version = 'bert-base-uncased'
model = BertForMaskedLM.from_pretrained(model_version)
model.eval()
cuda = torch.cuda.is_available()
if cuda:
    model = model.cuda()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))

CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    #print("input sent for detokenize: ",' '.join(sent))
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k 
    """
    #print(out)
    out = out["logits"]# the second of out tuple
    #print(type(out))
    #print(out.shape)
    #print(out)
    try:
        logits = out[:, gen_idx]
    except:
        print (gen_idx)
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx


def get_init_text(seed_text, max_len, batch_size = 1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    print("batch in get_init_text: ",batch)
    #if rand_init:
    #    for ii in range(max_len):
    #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))

    return tokenize_batch(batch)

def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    #print(" ".join(sent))
    return " ".join(sent)

# Generation modes as functions
import math

def parallel_sequential_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, burnin=200,
                                cuda=False, print_every=10, verbose=True):
    """ Generate for one random position at a timestep

    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    print("batch in parallel_sequential_generation: ",batch)

    for ii in range(max_iter):
        kk = np.random.randint(0, max_len)
        for jj in range(batch_size):
            batch[jj][seed_len+kk] = mask_id
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(out, gen_idx=seed_len+kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
        for jj in range(batch_size):
            batch[jj][seed_len+kk] = idxs[jj]
        
        if verbose and np.mod(ii+1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = for_print[:seed_len+kk+1] + ['(*)'] + for_print[seed_len+kk+1:]
            print("iter", ii+1, " ".join(for_print))
        
    return untokenize_batch(batch)

def generate(n_samples, seed_text="[CLS]", batch_size=10, max_len=25, 
            generation_mode="parallel-sequential",
            sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
            cuda=False, print_every=1):
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        if generation_mode == "parallel-sequential":
            batch = parallel_sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                                temperature=temperature, burnin=burnin, max_iter=max_iter, 
                                                cuda=cuda, verbose=False)
    
        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
    
        sentences += batch
        durationTime = time.time() - start_time
    return sentences, int(durationTime)

def get_model():
    def BertGenerator():
    
        n_samples = 1
        batch_size = 5
        max_len = 20
        top_k = 100 
        temperature = 1.0
        generation_mode = "parallel-sequential"
        leed_out_len = 5 # max_len
        burnin = 250
        sample = True
        max_iter = 500

        # add CLS
        original_text = "Everybody likes coffee."
        marked_text = "[CLS] " + original_text 
        # Tokenize our sentence with the BERT tokenizer.
        seed_text = tokenizer.tokenize(marked_text)
        #seed_text = ("[CLS] "+ original_text.lower()).split()
        print(seed_text)
        bert_sents, durationTime = generate(n_samples, seed_text=seed_text, batch_size=batch_size, max_len=max_len,
                            generation_mode=generation_mode,
                            sample=sample, top_k=top_k, temperature=temperature, burnin=burnin, max_iter=max_iter,
                            cuda=cuda)

        # for sent in bert_sents:
        #     printer(sent, should_detokenize=True)
        print(f"BERT finished in {int(durationTime)}")
            
        return printer(bert_sents[0], should_detokenize=True)
    
    return BertGenerator


BertGenerator = get_model()

from transformers import pipeline
def get_gpt3_model():
    def GPT3Generator():
        starttime = time.time()
        original_text = "Everybody likes coffee."
        max_len = 20
        GPT3= pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
        result = GPT3(original_text, max_length=max_len, do_sample=True, temperature=0.9)
        durationTime = time.time()- starttime
        print(f"GPT-3 finished in {int(durationTime)}")
            
        return result[0]['generated_text']
    
    return GPT3Generator


GPT3Generator = get_gpt3_model()

print('Model and data preloading completed in ', time.time()-start_time)
model1 = BertGenerator  
model2 = GPT3Generator


app = Flask(__name__)
app.config.update(dict(
    DEBUG=True,
    SECRET_KEY='MODEL_TESTING_BY_CHANG_LIU',
))


@app.route('/', methods=["GET"])
def main():
    # if no userid is defined make one up
    if 'userid' not in session:
        session['userid'] = random.randint(1,1000)

    model_perf_exp = ModelExperiment(userid=session['userid'])
    model_type = model_perf_exp.get('model_type')
    resp = {}
    resp["success"] = False

    print(model_type, resp, session['userid'])

    try:
        if model_type == 'BERT':
            generatedString = BertGenerator()
        elif model_type == 'GPT-3':
            generatedString = GPT3Generator()

        print(generatedString)
        resp["response"] = generatedString
        resp["success"] = True

        print(model_type, resp, session['userid'])

        return render_template_string("""
                <html>
                    <head>
                        <title>Text Generation Service</title>
                    </head>
                    <body>
                        <h3>
                            Generations for userid {{ userid }} based on {{ model_type }} are shown below: <br>
                        </h3>
                        <p>
                        
                              <h5> {{resp['response']}}</h5>
                        
                        </p>
                        <p>
                            What will be your rating of this generation (rate between 1-10 where 10 is the highest quality)?
                        </p>
                        <form action="/rate" method="GET">
                            <input type="text" length="10" name="rate"></input>
                            <input type="submit"></input>
                        </form>
                    <br>
                    <p><a href="/">Reload without resetting my user ID. I'll get the same generator when I come back.</a></p>
                    <p><a href="/reset">Reset my user ID so I am a different user and will get re-randomized into a new treatment.</a></p>
                    </body>
                </html>
            """, userid=session['userid'], model_type=model_type, resp=resp)
    except:
        return render_template_string("""
            <html>
                <head>
                    <title>Text Generation Service</title>
                </head>
                <body>
                    <h3>
                        Generations for userid {{ userid }} based on {{ model_type }} are shown below. <br>
                    </h3>
                    <p>
                    {{resp}}
                    </p>
                    <p>
                        What will be your rating of this list (rate between 1-10 where 10 is the highest quality)?
                    </p>
                    <form action="/rate" method="GET">
                        <input type="text" length="10" name="rate"></input>
                        <input type="submit"></input>
                    </form>
                <br>
                <p><a href="/">Reload without resetting my user ID. I'll get the same recommendations when I come back.</a></p>
                <p><a href="/reset">Reset my user ID so I am a different user and will get re-randomized into a new treatment.</a></p>
                </body>
            </html>
            """, userid=session['userid'], model_type=model_type, resp=resp)


@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('main'))


@app.route('/rate')
def rate():
    rate_string = request.args.get('rate')
    try:
        rate_val = int(rate_string)
        assert rate_val > 0 and rate_val < 11

        model_perf_exp = ModelExperiment(userid=session['userid'])
        model_perf_exp.log_event('rate', {'rate_val': rate_val})

        return render_template_string("""
                    <html>
                        <head>
                            <title>Thank you for the feedback!</title>
                        </head>
                        <body>
                            <p>You rating is {{ rate_val }}. Hit the back button or click below to go back to generators!</p>
                            <p><a href="/">Back</a></p>
                        </body>
                    </html>
                    """, rate_val=rate_val)
    except:
        return render_template_string("""
                    <html>
                        <head>
                            <title>Bad rating!</title>
                        </head>
                        <body>
                            <p>You rating could not be parsed. That's probably not a number between 1 and 10, so we won't be accepting your rating.</p>
                            <p><a href="/">Back</a></p>
                        </body>
                    </html>
                    """)


# start the flask app, allow remote connections
app.run(host='0.0.0.0')