#!/usr/bin/env python
from optparse import OptionParser
import os
import sys

import h5py
import numpy as np

'''
w5_merge.py

Merge wig5 files using a specified summary statistic.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <out_w5> <in1_w5> <in2_w5> ...'
    parser = OptionParser(usage)
    parser.add_option('-s', dest='sum_stat',
        default='sum', help='Summary statistic [Default: %default]')
    parser.add_option('-v', dest='verbose',
        default=False, action='store_true')
    parser.add_option('-w', dest='overwrite',
        default=False, action='store_true')
    parser.add_option('-z', dest='gzip',
        default=False, action='store_true')
    (options,args) = parser.parse_args()

    if len(args) < 3:
        parser.error('Must provide output and two or more input wig5.')
    else:
        out_w5_file = args[0]
        in_w5_files = args[1:]

    compression_args = {}
    if options.gzip:
        compression_args['compression'] = 'gzip'
        compression_args['shuffle'] = True

    # open input wig5
    in_w5_opens = [h5py.File(iwf) for iwf in in_w5_files]
    in_num = len(in_w5_opens)

    # take keys union
    in_keys = set()
    for in_w5_open in in_w5_opens:
        in_keys |= in_w5_open.keys()

    # open output file
    if os.path.isfile(out_w5_file) and not options.overwrite:
        parser.error('%s exists. Please remove.' % out_w5_file)
    out_w5_open = h5py.File(out_w5_file, 'w')

    for out_key in in_keys:
        if options.verbose:
            print(out_key)

        # initialize array
        for i in range(in_num):
            if out_key in in_w5_opens[i]:
                in_key_len = len(in_w5_opens[i][out_key])
                break
        in_key_data = np.zeros((in_num,in_key_len), dtype='float32')

        # read data
        for i in range(in_num):
            if out_key in in_w5_opens[i]:
                in_key_data[i] = np.array(in_w5_opens[i][out_key])
            else:
                print('%s missing %s' % (in_w5_files[i], out_key), file=sys.stderr)

        # summarize
        if options.sum_stat == 'sum':
            out_key_data = in_key_data.sum(axis=0)

        elif options.sum_stat == 'mean':
            out_key_data = in_key_data.mean(axis=0)

        elif options.sum_stat == 'geo-mean':
            in_key_data_log = np.log(in_key_data)
            in_key_data_log_mean = in_key_data_log.mean(axis=0)
            out_key_data = np.exp(in_key_data_log_mean)

        elif options.sum_stat == 'sqrt-mean':
            in_key_data_sqrt = in_key_data**0.5
            in_key_data_sqrt_mean = in_key_data_sqrt.mean(axis=0)
            out_key_data = in_key_data_sqrt_mean**2
            
        else:
            print('Cannot identify summary statistic %s' % options.sum_stat)

        # carefully decrease resolution
        out_key_data = np.clip(out_key_data, np.finfo(np.float16).min, np.finfo(np.float16).max)
        out_key_data = out_key_data.astype('float16')

        # write
        out_w5_open.create_dataset(out_key, data=out_key_data,
                                   dtype='float16', **compression_args)

    out_w5_open.close()



################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
