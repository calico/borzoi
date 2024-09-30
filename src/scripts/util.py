#!/usr/bin/env python
from __future__ import print_function
#import pdb
import operator, os, sys, subprocess, time

############################################################
# util
#
# Helpful methods that are difficult to categorize.
############################################################

############################################################
# condorify
############################################################
def condorify(cmds):
    return ['runCmd -c "%s"' % c for c in cmds]

############################################################
# slurmify
############################################################
def slurmify(cmds, mem_mb=None):
    if mem != None:
        mem_str = '--mem %d' % mem_mb
    else:
        mem_str = ''

    return ['srun -p general -n 1 %s "%s"' % (mem_str,c) for c in cmds]

############################################################
# exec_par
#
# Execute the commands in the list 'cmds' in parallel, but
# only running 'max_proc' at a time.
############################################################
def exec_par(cmds, max_proc=None, verbose=False):
    total = len(cmds)
    finished = 0
    running = 0
    p = []

    if max_proc == None:
        max_proc = len(cmds)

    if max_proc == 1:
        while finished < total:
            if verbose:
                print(cmds[finished], file=sys.stderr)
            op = subprocess.Popen(cmds[finished], shell=True)
            os.waitpid(op.pid, 0)
            finished += 1

    else:
        while finished + running < total:
            # launch jobs up to max
            while running < max_proc and finished+running < total:
                if verbose:
                    print(cmds[finished+running], file=sys.stderr)
                p.append(subprocess.Popen(cmds[finished+running], shell=True))
                # print('Running %d' % p[running].pid)
                running += 1

            # are any jobs finished
            new_p = []
            for i in range(len(p)):
                # print('POLLING', i, p[i].poll())
                if p[i].poll() != None:
                    running -= 1
                    finished += 1
                else:
                    new_p.append(p[i])

            # if none finished, sleep
            if len(new_p) == len(p):
                time.sleep(1)
            p = new_p

        # wait for all to finish
        for i in range(len(p)):
            p[i].wait()

############################################################
# slurm_par
#
# Execute the commands in the list 'cmds' in parallel on
# SLURM, but only running 'max_proc' at a time.
#
# Doesn't work. Jobs are allocated resources, but won't run.
# Also, I'd have to screen into login nodes, which
# isn't great because I can't get back to them.
############################################################
def slurm_par(cmds, max_proc, queue='general', cpu=1, mem=None, out_files=None, err_files=None):
    # preprocess cmds
    if mem != None:
        mem_str = '--mem %d' % mem
    else:
        mem_str = ''

    if out_files != None:
        out_strs = ['-o %s' % of for of in out_files]
    else:
        out_strs = ['']*len(cmds)

    if err_files != None:
        err_strs = ['-e %s' % ef for ef in err_files]
    else:
        err_strs = ['']*len(cmds)

    slurm_cmds = ['srun -p %s -n %d %s %s %s "%s"' % (queue, cpu, mem_str, out_strs[i], err_strs[i], cmds[i]) for i in range(len(cmds))]

    exec_par(slurm_cmds, max_proc, print_cmd=True)


############################################################
# sort_dict
#
# Sort a dict by the values, returning a list of tuples
############################################################
def sort_dict(hash, reverse=False):
    return sorted(hash.items(), key=operator.itemgetter(1), reverse=reverse)

