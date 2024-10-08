#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import os, pdb, sys, subprocess, tempfile, time

################################################################################
# slurm.py
#
# Methods to run jobs on SLURM.
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('-g', dest='go',
            default=False, action='store_true',
            help='Don\'t wait for the job to finish [Default: %default]')

    parser.add_option('-o', dest='out_file')
    parser.add_option('-e', dest='err_file')

    parser.add_option('-J', dest='job_name')

    parser.add_option('-q', dest='queue', default='general')
    parser.add_option('-n', dest='cpu', default=1, type='int')
    parser.add_option('-m', dest='mem', default=None, type='int')
    parser.add_option('-t', dest='time', default=None)

    (options,args) = parser.parse_args()

    cmd = args[0]

    main_job = Job(cmd, name=options.job_name,
        out_file=options.out_file, err_file=options.err_file,
        queue=options.queue, cpu=options.cpu,
        mem=options.mem, time=options.time)
    main_job.launch()

    if options.go:
        time.sleep(1)

        # find the job
        if not main_job.update_status:
            time.sleep(1)

        # delete sbatch
        main_job.clean()

    else:
        time.sleep(10)

        # find the job
        if not main_job.update_status():
            time.sleep(10)

        # wait for it to complete
        while main_job.update_status() and main_job.status in ['PENDING','RUNNING']:
            time.sleep(30)

        print('%s %s' % (main_job.name, main_job.status), file=sys.stderr)

        # delete sbatch
        main_job.clean()


################################################################################
# multi_run
#
# Launch and manage multiple SLURM jobs in parallel, using only one 'sacct'
# call per
################################################################################
def multi_run(jobs, max_proc=None, verbose=False, launch_sleep=2, update_sleep=20):
    total = len(jobs)
    finished = 0
    running = 0
    active_jobs = []

    if max_proc is None:
        max_proc = len(jobs)

    while finished + running < total:
        # launch jobs up to the max
        while running < max_proc and finished+running < total:
            # launch
            jobs[finished+running].launch()
            time.sleep(launch_sleep)
            if verbose:
                print(jobs[finished+running].name, jobs[finished+running].cmd, file=sys.stderr)

            # save it
            active_jobs.append(jobs[finished+running])
            running += 1

        # sleep
        time.sleep(update_sleep)

        # update all statuses
        multi_update_status(active_jobs)

        # update active jobs
        active_jobs_new = []
        for i in range(len(active_jobs)):
            if active_jobs[i].status in ['PENDING', 'RUNNING']:
                active_jobs_new.append(active_jobs[i])
            else:
                if verbose:
                    print('%s %s' % (active_jobs[i].name, active_jobs[i].status), file=sys.stderr)

                running -= 1
                finished += 1

        active_jobs = active_jobs_new


    # wait for all to finish
    while active_jobs:
        # sleep
        time.sleep(update_sleep)

        # update all statuses
        multi_update_status(active_jobs)

        # update active jobs
        active_jobs_new = []
        for i in range(len(active_jobs)):
            if active_jobs[i].status in ['PENDING', 'RUNNING']:
                active_jobs_new.append(active_jobs[i])
            else:
                if verbose:
                    print('%s %s' % (active_jobs[i].name, active_jobs[i].status), file=sys.stderr)

                running -= 1
                finished += 1

        active_jobs = active_jobs_new


################################################################################
# multi_update_status
#
# Update the status for multiple jobs at once.
################################################################################
def multi_update_status(jobs, max_attempts=3, sleep_attempt=5):
    # reset all
    for j in jobs:
        j.status = None

    # try multiple times because sometimes it fails
    attempt = 0
    while attempt < max_attempts and [j for j in jobs if j.status == None]:
        if attempt > 0:
            time.sleep(sleep_attempt)

        sacct_str = subprocess.check_output('sacct', shell=True)
        sacct_str = sacct_str.decode('UTF-8')

        # split into job lines
        sacct_lines = sacct_str.split('\n')
        for line in sacct_lines[2:]:
            a = line.split()

            try:
                line_id = int(a[0])
            except:
                line_id = None

            # check call jobs for a match
            for j in jobs:
                if line_id == j.id:
                    # j.status = a[5] # original
                    j.status = a[4] # cb2

        attempt += 1


class Job:
    ''' class to manage SLURM jobs.

    Notes:
     -Since we have two types of machines in the GPU queue, I'm asking
      for the machine type as "queue", and the "launch" method will handle it.
    '''

    def __init__(self, cmd, name, out_file=None, err_file=None, sb_file=None,
                 queue='standard', cpu=1, mem=None, time=None, gpu=0):
        self.cmd = cmd
        self.name = name
        self.out_file = out_file
        self.err_file = err_file
        self.sb_file = sb_file
        self.queue = self.translate_gpu(queue)
        self.cpu = cpu
        self.mem = mem
        self.time = time
        self.gpu = gpu

        self.id = None
        self.status = None


    def flash(self):
        ''' Determine if the job can run on the flash queue by parsing the time. '''

        day_split = self.time.split('-')
        if len(day_split) == 2:
            days, hms = day_split
        else:
            days = 0
            hms = day_split[0]

        hms_split = hms.split(':')
        if len(hms_split) == 3:
            hours, mins, secs = hms_split
        elif len(hms_split) == 2:
            hours = 0
            mins, secs = hms_split
        else:
            print('Cannot parse time: ', self.time, file=sys.stderr)
            exit(1)

        hours_sum = 24*int(days) + int(hours) + float(mins)/60

        return hours_sum <= 4


    def launch(self):
        ''' Make an sbatch file, launch it, and save the job id. '''

        # make sbatch script
        if self.sb_file is None:
            sbatch_tempf = tempfile.NamedTemporaryFile()
            sbatch_file = sbatch_tempf.name
        else:
            sbatch_file = self.sb_file
        sbatch_out = open(sbatch_file, 'w')

        print('#!/bin/bash\n', file=sbatch_out)
        if self.gpu > 0:
            if self.queue == "" or self.queue == 'gpu':
                gpu_str = 'gpu'
                gres_str = '--gres=gpu'
            elif self.queue == 'nvidia_geforce_rtx_4090':
                gpu_str = 'minigpu'
                gres_str = '--gres=gpu:%s' % self.queue
            else:
                gpu_str = 'gpu'
                gres_str = '--gres=gpu:%s' % self.queue
            print('#SBATCH -p %s' % gpu_str, file=sbatch_out)
            print('#SBATCH %s:%d\n' % (gres_str, self.gpu), file=sbatch_out)
        else:
            print('#SBATCH -p %s' % self.queue, file=sbatch_out)
        print('#SBATCH -n 1', file=sbatch_out)
        print('#SBATCH -c %d' % self.cpu, file=sbatch_out)
        if self.name:
            print('#SBATCH -J %s' % self.name, file=sbatch_out)
        if self.out_file:
            print('#SBATCH -o %s' % self.out_file, file=sbatch_out)
        if self.err_file:
            print('#SBATCH -e %s' % self.err_file, file=sbatch_out)
        if self.mem:
            print('#SBATCH --mem %d' % self.mem, file=sbatch_out)
        if self.time:
            print('#SBATCH --time %s' % self.time, file=sbatch_out)
        print(self.cmd, file=sbatch_out)

        sbatch_out.close()

        # launch it; check_output to get the id
        launch_str = subprocess.check_output('sbatch %s' % sbatch_file, shell=True)

        # e.g. "Submitted batch job 13861989"
        self.id = int(launch_str.split()[3])


    def translate_gpu(self, queue_gpu):
        """Translate concise GPU labels to their full versions,
            or propagate the given label."""
        translation = {
            'p100': 'tesla_p100-pcie-16gb',
            'tesla': 'tesla_p100-pcie-16gb',
            'geforce': 'nvidia_geforce_gtx_1080_ti',
            'gtx1080': 'nvidia_geforce_gtx_1080_ti',
            'titan': 'titan_rtx',
            'quadro': 'quadro_rtx_8000',
            'rtx4090': 'nvidia_geforce_rtx_4090'
        }
        return translation.get(queue_gpu, queue_gpu)


    def update_status(self, max_attempts=3, sleep_attempt=5):
        ''' Use 'sacct' to update the job's status. Return True if found and False if not. '''

        status = None

        attempt = 0
        while attempt < max_attempts and status == None:
            if attempt > 0:
                time.sleep(sleep_attempt)

            sacct_str = subprocess.check_output('sacct', shell=True)
            sacct_str = sacct_str.decode('UTF-8')

            sacct_lines = sacct_str.split('\n')
            for line in sacct_lines[2:]:
                a = line.split()

                try:
                    line_id = int(a[0])
                except:
                    line_id = None

                if line_id == self.id:
                    status = a[5]

            attempt += 1

        if status == None:
            return False
        else:
            self.status = status
            return True


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()