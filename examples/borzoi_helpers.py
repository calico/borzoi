import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

import baskerville
from baskerville import seqnn
from baskerville import dna

import pysam

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

import intervaltree
import pyBigWig

import gc

# Helper functions for prediction, attribution, and visualization

# Make one-hot coded sequence
def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)

    # Extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    seq_1hot = dna.dna_1hot(seq_dna)
    return seq_1hot


# Predict coverage tracks
def predict_tracks(models, sequence_one_hot):

    predicted_tracks = []
    
    #Loop over model replicates
    for rep_ix in range(len(models)):

        #Predict coverage and store as float16
        yh = models[rep_ix](sequence_one_hot[None, ...])[:, None, ...].astype(
            "float16"
        )

        predicted_tracks.append(yh)

    #Concatenate across replicates
    predicted_tracks = np.concatenate(predicted_tracks, axis=1)

    return predicted_tracks


# Helper function to get (padded) one-hot
def process_sequence(fasta_open, chrom, start, end, seq_len=524288):

    seq_len_actual = end - start

    # Pad sequence to input window size
    start -= (seq_len - seq_len_actual) // 2
    end += (seq_len - seq_len_actual) // 2

    # Get one-hot
    sequence_one_hot = make_seq_1hot(fasta_open, chrom, start, end, seq_len)

    return sequence_one_hot.astype("float32")


#Function to plot a DNA letter at a specified coordinate in a subplot axis
def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):

    fp = FontProperties(family="DejaVu Sans", weight="bold")

    globscale = 1.35

    #Letter graphics parameters
    LETTERS = {
        "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
        "UP": TextPath((-0.488, 0), "$\\Uparrow$", size=1, prop=fp),
        "DN": TextPath((-0.488, 0), "$\\Downarrow$", size=1, prop=fp),
        "(": TextPath((-0.25, 0), "(", size=1, prop=fp),
        ".": TextPath((-0.125, 0), "-", size=1, prop=fp),
        ")": TextPath((-0.1, 0), ")", size=1, prop=fp),
    }

    #Letter colors
    COLOR_SCHEME = {
        "G": "orange",
        "A": "green",
        "C": "blue",
        "T": "red",
        "UP": "green",
        "DN": "red",
        "(": "black",
        ".": "black",
        ")": "black",
    }

    text = LETTERS[letter]

    #Optionally override default color
    chosen_color = COLOR_SCHEME[letter]
    if color is not None:
        chosen_color = color

    #Calculate transformed coordinates
    t = (
        mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale)
        + mpl.transforms.Affine2D().translate(x, y)
        + ax.transData
    )
    
    #Draw patch
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)

    #Add patch into axis subplot
    if ax != None:
        ax.add_artist(p)
    
    return p


#Tensorflow helper function to compute gradient of a given statistic predicted by the model
def _prediction_input_grad(
    input_sequence,
    model,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft,
    use_mean,
    use_ratio,
    use_logodds,
    subtract_avg,
    prox_bin_index,
    dist_bin_index,
    untransform_old,
):

    mean_dist_prox_ratio = None
    with tf.GradientTape() as tape:
        tape.watch(input_sequence)

        #Predict coverage for chosen tracks
        preds = tf.gather(
            model(input_sequence, training=False),
            tf.tile(
                tf.constant(np.array(track_index))[None, :],
                (tf.shape(input_sequence)[0], 1),
            ),
            axis=2,
            batch_dims=1,
        )

        #Undo transformations
        if untransform_old :
            
            #Undo scale
            preds = preds / track_scale

            #Undo clip-soft
            if clip_soft is not None:
                preds = tf.where(
                    preds > clip_soft, (preds - clip_soft) ** 2 + clip_soft, preds
                )

            #Undo sqrt
            preds = preds ** (1. / track_transform)
        else :
            
            #Undo clip-soft
            if clip_soft is not None :
                preds = tf.where(
                    preds > clip_soft, (preds - clip_soft + 1)**2 + clip_soft - 1, preds
                )

            #Undo sqrt
            preds = (preds + 1)**(1. / track_transform) - 1

            #Undo scale
            preds = preds / track_scale

        #Aggregate over tracks (average)
        pred = tf.reduce_mean(preds, axis=2)

        #Aggregate coverage across positions
        if not use_mean:
            #Sum over a range or an array of bins (distal)
            if dist_bin_index is None:
                mean_dist = tf.reduce_sum(pred[:, dist_bin_start:dist_bin_end], axis=1)
            else:
                mean_dist = tf.reduce_sum(
                    tf.gather(pred, dist_bin_index, axis=1), axis=1
                )
            
            #Sum over a range or an array of bins (proximal)
            if prox_bin_index is None:
                mean_prox = tf.reduce_sum(pred[:, prox_bin_start:prox_bin_end], axis=1)
            else:
                mean_prox = tf.reduce_sum(
                    tf.gather(pred, prox_bin_index, axis=1), axis=1
                )
        else:
            #Average over a range or an array of bins (distal)
            if dist_bin_index is None:
                mean_dist = tf.reduce_mean(pred[:, dist_bin_start:dist_bin_end], axis=1)
            else:
                mean_dist = tf.reduce_mean(
                    tf.gather(pred, dist_bin_index, axis=1), axis=1
                )
            
            #Average over a range or an array of bins (proximal)
            if prox_bin_index is None:
                mean_prox = tf.reduce_mean(pred[:, prox_bin_start:prox_bin_end], axis=1)
            else:
                mean_prox = tf.reduce_mean(
                    tf.gather(pred, prox_bin_index, axis=1), axis=1
                )
        
        #Apply a log transform (or a log ratio transform)
        if not use_ratio:
            mean_dist_prox_ratio = tf.math.log(mean_dist + 1e-6)
        else:
            #Apply a log ratio or log odds ratio transform
            if not use_logodds:
                mean_dist_prox_ratio = tf.math.log(mean_dist / mean_prox + 1e-6)
            else:
                mean_dist_prox_ratio = tf.math.log(
                    (mean_dist / mean_prox) / (1. - (mean_dist / mean_prox)) + 1e-6
                )

    #Get the gradient and mean-subtract the result
    input_grad = tape.gradient(mean_dist_prox_ratio, input_sequence)
    if subtract_avg:
        input_grad = input_grad - tf.reduce_mean(input_grad, axis=-1, keepdims=True)
    else:
        input_grad = input_grad

    return input_grad


#Function to compute the average input gradient for the sequence and its reverse-complement
def get_prediction_gradient_w_rc(
    models,
    sequence_one_hots,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft=None,
    prox_bin_index=None,
    dist_bin_index=None,
    use_mean=False,
    use_ratio=True,
    use_logodds=False,
    subtract_avg=False,
    fold_index=[0, 1, 2, 3],
    untransform_old=False,
):

    #Get gradients for sequence
    pred_grads = get_prediction_gradient(
        models,
        sequence_one_hots,
        prox_bin_start,
        prox_bin_end,
        dist_bin_start,
        dist_bin_end,
        track_index,
        track_scale,
        track_transform,
        clip_soft,
        prox_bin_index,
        dist_bin_index,
        use_mean,
        use_ratio,
        use_logodds,
        subtract_avg,
        fold_index,
        untransform_old,
    )

    #Get reverse-complemented sequence
    sequence_one_hots_rc = [
        sequence_one_hots[example_ix][::-1, ::-1]
        for example_ix in range(len(sequence_one_hots))
    ]

    #Get reverse-complemented positions
    prox_bin_start_rc = models[0].target_lengths[0] - prox_bin_start - 1
    prox_bin_end_rc = models[0].target_lengths[0] - prox_bin_end - 1

    dist_bin_start_rc = models[0].target_lengths[0] - dist_bin_start - 1
    dist_bin_end_rc = models[0].target_lengths[0] - dist_bin_end - 1

    #Reverse-complement position indices (if they are given as arguments); proximal
    prox_bin_index_rc = None
    if prox_bin_index is not None:
        prox_bin_index_rc = [
            models[0].target_lengths[0] - prox_bin - 1 for prox_bin in prox_bin_index
        ]

    #Reverse-complement position indices (if they are given as arguments); distal
    dist_bin_index_rc = None
    if dist_bin_index is not None:
        dist_bin_index_rc = [
            models[0].target_lengths[0] - dist_bin - 1 for dist_bin in dist_bin_index
        ]

    #Get gradients for reverse-complemented sequence
    pred_grads_rc = get_prediction_gradient(
        models,
        sequence_one_hots_rc,
        prox_bin_end_rc,
        prox_bin_start_rc,
        dist_bin_end_rc,
        dist_bin_start_rc,
        track_index,
        track_scale,
        track_transform,
        clip_soft,
        prox_bin_index_rc,
        dist_bin_index_rc,
        use_mean,
        use_ratio,
        use_logodds,
        subtract_avg,
        fold_index,
        untransform_old,
    )

    #Average gradient saliencies
    pred_grads_avg = [
        (pred_grads[example_ix] + pred_grads_rc[example_ix][::-1, ::-1]) / 2.
        for example_ix in range(len(sequence_one_hots))
    ]

    return pred_grads, pred_grads_rc, pred_grads_avg


#Function to compute input-gated, mean-subtracted gradient saliencies for a list of sequences
def get_prediction_gradient(
    models,
    sequence_one_hots,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft=None,
    prox_bin_index=None,
    dist_bin_index=None,
    use_mean=False,
    use_ratio=True,
    use_logodds=False,
    subtract_avg=False,
    fold_index=[0, 1, 2, 3],
    untransform_old=False,
):

    #Initialize structure to record gradients for multiple model replicates
    pred_grads = np.zeros((len(sequence_one_hots), len(fold_index), 524288, 4))

    #Loop over model replicates
    for fold_i, fold_ix in enumerate(fold_index) :

        #Get model
        prediction_model = models[fold_ix].model.layers[1]

        #Initialize new keras input layer
        input_sequence = tf.keras.layers.Input(shape=(524288, 4), name="sequence")

        #Make a lambda layer with the gradient statistic tensorflow function
        input_grad = tf.keras.layers.Lambda(
            lambda x: _prediction_input_grad(
                x,
                prediction_model,
                prox_bin_start,
                prox_bin_end,
                dist_bin_start,
                dist_bin_end,
                track_index,
                track_scale,
                track_transform,
                clip_soft,
                use_mean,
                use_ratio,
                use_logodds,
                subtract_avg,
                prox_bin_index,
                dist_bin_index,
                untransform_old,
            ),
            name="inp_grad",
        )(input_sequence)

        #Compile a new model to calculate the gradient
        grad_model = tf.keras.models.Model(input_sequence, input_grad)

        #Run gradient calculation on CPU
        with tf.device("/cpu:0"):
            
            #Loop over sequences
            for example_ix in range(len(sequence_one_hots)) :
                
                #Calculate and store input-gated gradient
                pred_grads[example_ix, fold_i, ...] = (
                    sequence_one_hots[example_ix]
                    * grad_model.predict(
                        x=[sequence_one_hots[example_ix][None, ...]],
                        batch_size=1,
                        verbose=True,
                    )[0, ...]
                )

        #Run garbage collection before next gradient computation
        prediction_model = None
        gc.collect()

    #Average across model replications
    pred_grads = np.mean(pred_grads, axis=1)
    
    #Project to nucleotides again
    pred_grads = [
        np.sum(pred_grads[example_ix, ...], axis=-1, keepdims=True)
        * sequence_one_hots[example_ix]
        for example_ix in range(len(sequence_one_hots))
    ]

    return pred_grads


#Helper function to compute summary statistic from predicted coverage track
def _prediction_ism_score(
    pred,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    use_mean,
    use_ratio,
    use_logodds,
    prox_bin_index,
    dist_bin_index,
):

    #Aggregate across positions
    if not use_mean:
        #Sum over a range or an array of positions (distal)
        if dist_bin_index is None:
            mean_dist = np.sum(pred[:, dist_bin_start:dist_bin_end], axis=1)
        else:
            mean_dist = np.sum(pred[:, dist_bin_index], axis=1)
        
        #Sum over a range or an array of positions (proximal)
        if prox_bin_index is None:
            mean_prox = np.sum(pred[:, prox_bin_start:prox_bin_end], axis=1)
        else:
            mean_prox = np.sum(pred[:, prox_bin_index], axis=1)
    else:
        
        #Average over a range or an array of positions (distal)
        if dist_bin_index is None:
            mean_dist = np.mean(pred[:, dist_bin_start:dist_bin_end], axis=1)
        else:
            mean_dist = np.mean(pred[:, dist_bin_index], axis=1)
        
        
        #Average over a range or an array of positions (proximal)
        if prox_bin_index is None:
            mean_prox = np.mean(pred[:, prox_bin_start:prox_bin_end], axis=1)
        else:
            mean_prox = np.mean(pred[:, prox_bin_index], axis=1)

    #Apply a log transform (or a log ratio transform)
    if not use_ratio:
        mean_dist_prox_ratio = np.log(mean_dist + 1e-6)
    else:
        #Apply a log ratio or log odds ratio transform
        if not use_logodds:
            mean_dist_prox_ratio = np.log(mean_dist / mean_prox + 1e-6)
        else:
            mean_dist_prox_ratio = np.log(
                (mean_dist / mean_prox) / (1. - (mean_dist / mean_prox)) + 1e-6
            )

    return mean_dist_prox_ratio


#Function to compute ISM maps for a list of sequences
def get_ism(
    models,
    sequence_one_hots,
    ism_start,
    ism_end,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft,
    prox_bin_index=None,
    dist_bin_index=None,
    use_mean=False,
    use_ratio=True,
    use_logodds=False,
    untransform_old=False,
):

    #Initialize array to store ISM results across model replicates
    pred_ism = np.zeros((len(sequence_one_hots), len(models), 524288, 4))

    bases = [0, 1, 2, 3]

    #Loop over sequences
    for example_ix in range(len(sequence_one_hots)):

        print("example_ix = " + str(example_ix))

        sequence_one_hot_wt = sequence_one_hots[example_ix]

        #Get pred
        y_wt = predict_tracks(models, sequence_one_hot_wt)[0, ...][
            ..., track_index
        ].astype("float32")

        #Undo transforms
        
        if untransform_old :
            
            #Undo scale
            y_wt /= track_scale

            #Undo clip-soft
            if clip_soft is not None:
                y_wt_unclipped = (y_wt - clip_soft) ** 2 + clip_soft
                unclip_mask_wt = y_wt > clip_soft

                y_wt[unclip_mask_wt] = y_wt_unclipped[unclip_mask_wt]

            #Undo sqrt
            y_wt = y_wt ** (1. / track_transform)
        else :
            
            #Undo clip-soft
            if clip_soft is not None :
                y_wt_unclipped = (y_wt - clip_soft + 1)**2 + clip_soft - 1
                unclip_mask_wt = (y_wt > clip_soft)

                y_wt[unclip_mask_wt] = y_wt_unclipped[unclip_mask_wt]

            #Undo sqrt
            y_wt = (y_wt + 1)**(1. / track_transform) - 1

            #Undo scale
            y_wt /= track_scale

        #Aggregate over tracks (average)
        y_wt = np.mean(y_wt, axis=-1)

        #Calculate reference statistic
        score_wt = _prediction_ism_score(
            y_wt,
            prox_bin_start,
            prox_bin_end,
            dist_bin_start,
            dist_bin_end,
            use_mean,
            use_ratio,
            use_logodds,
            prox_bin_index,
            dist_bin_index,
        )

        #Loop over ISM positions
        for j in range(ism_start, ism_end):
            
            #Loop over nucleotides
            for b in bases:
                
                #Calculate ISM score if nucleotide is different from reference
                if sequence_one_hot_wt[j, b] != 1.:
                    
                    #Copy sequence and induce mutation
                    sequence_one_hot_mut = np.copy(sequence_one_hot_wt)
                    sequence_one_hot_mut[j, :] = 0.
                    sequence_one_hot_mut[j, b] = 1.

                    #Get pred
                    y_mut = predict_tracks(models, sequence_one_hot_mut)[0, ...][
                        ..., track_index
                    ].astype("float32")

                    #Undo transforms
                    
                    if untransform_old :
                        #Undo scale
                        y_mut /= track_scale

                        #Undo clip-soft
                        if clip_soft is not None:
                            y_mut_unclipped = (y_mut - clip_soft) ** 2 + clip_soft
                            unclip_mask_mut = y_mut > clip_soft

                            y_mut[unclip_mask_mut] = y_mut_unclipped[unclip_mask_mut]

                        #Undo sqrt
                        y_mut = y_mut ** (1. / track_transform)
                    else :
                        #Undo clip-soft
                        if clip_soft is not None :
                            y_mut_unclipped = (y_mut - clip_soft + 1)**2 + clip_soft - 1
                            unclip_mask_mut = (y_mut > clip_soft)

                            y_mut[unclip_mask_mut] = y_mut_unclipped[unclip_mask_mut]

                        #Undo sqrt
                        y_mut = (y_mut + 1)**(1. / track_transform) - 1

                        #Undo scale
                        y_mut /= track_scale

                    #Aggregate over tracks (average)
                    y_mut = np.mean(y_mut, axis=-1)

                    #Calculate variant statistic
                    score_mut = _prediction_ism_score(
                        y_mut,
                        prox_bin_start,
                        prox_bin_end,
                        dist_bin_start,
                        dist_bin_end,
                        use_mean,
                        use_ratio,
                        use_logodds,
                        prox_bin_index,
                        dist_bin_index,
                    )

                    pred_ism[example_ix, :, j, b] = score_wt - score_mut

        #Average across mutations per positions and broadcast back to nucleotides
        pred_ism[example_ix, ...] = (
            np.tile(np.mean(pred_ism[example_ix, ...], axis=-1)[..., None], (1, 1, 4))
            * sequence_one_hots[example_ix][None, ...]
        )

    #Average across model replicates
    pred_ism = np.mean(pred_ism, axis=1)
    pred_ism = [
        pred_ism[example_ix, ...] for example_ix in range(len(sequence_one_hots))
    ]

    return pred_ism


#Function to compute ISM Shuffle maps for a list of sequences
def get_ism_shuffle(
    models,
    sequence_one_hots,
    ism_start,
    ism_end,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft,
    prox_bin_index=None,
    dist_bin_index=None,
    window_size=5,
    n_samples=8,
    mononuc_shuffle=False,
    dinuc_shuffle=False,
    use_mean=False,
    use_ratio=True,
    use_logodds=False,
    untransform_old=False,
):

    #Initialize array to store shuffle results across model replicates
    pred_shuffle = np.zeros((len(sequence_one_hots), len(models), 524288, n_samples))
    pred_ism = np.zeros((len(sequence_one_hots), len(models), 524288, 4))

    bases = [0, 1, 2, 3]

    #Loop over sequences
    for example_ix in range(len(sequence_one_hots)):

        print("example_ix = " + str(example_ix))

        sequence_one_hot_wt = sequence_one_hots[example_ix]

        #Get pred
        y_wt = predict_tracks(models, sequence_one_hot_wt)[0, ...][
            ..., track_index
        ].astype("float32")

        #Undo transforms
        
        if untransform_old :
            
            #Undo scale
            y_wt /= track_scale

            #Undo clip-soft
            if clip_soft is not None:
                y_wt_unclipped = (y_wt - clip_soft) ** 2 + clip_soft
                unclip_mask_wt = y_wt > clip_soft

                y_wt[unclip_mask_wt] = y_wt_unclipped[unclip_mask_wt]

            #Undo sqrt
            y_wt = y_wt ** (1. / track_transform)
        else :
            
            #Undo clip-soft
            if clip_soft is not None :
                y_wt_unclipped = (y_wt - clip_soft + 1)**2 + clip_soft - 1
                unclip_mask_wt = (y_wt > clip_soft)

                y_wt[unclip_mask_wt] = y_wt_unclipped[unclip_mask_wt]

            #Undo sqrt
            y_wt = (y_wt + 1)**(1. / track_transform) - 1

            #Undo scale
            y_wt /= track_scale

        #Aggregate over tracks (average)
        y_wt = np.mean(y_wt, axis=-1)

        #Calculate reference statistic
        score_wt = _prediction_ism_score(
            y_wt,
            prox_bin_start,
            prox_bin_end,
            dist_bin_start,
            dist_bin_end,
            use_mean,
            use_ratio,
            use_logodds,
            prox_bin_index,
            dist_bin_index,
        )

        #Loop over shuffle positions
        for j in range(ism_start, ism_end):

            #Calculate local window positions (to shuffle)
            j_start = j - window_size // 2
            j_end = j + window_size // 2 + 1

            pos_index = np.arange(j_end - j_start) + j_start

            #Loop over the number of independent shuffle samples
            for sample_ix in range(n_samples):
                sequence_one_hot_mut = np.copy(sequence_one_hot_wt)
                sequence_one_hot_mut[j_start:j_end, :] = 0.

                #Randomly mutate or mono-nucleotide-shuffle
                if not mononuc_shuffle and not dinuc_shuffle:
                    nt_index = np.random.choice(bases, size=(j_end - j_start,)).tolist()
                    sequence_one_hot_mut[pos_index, nt_index] = 1.
                elif mononuc_shuffle:
                    shuffled_pos_index = np.copy(pos_index)
                    np.random.shuffle(shuffled_pos_index)

                    sequence_one_hot_mut[shuffled_pos_index, :] = sequence_one_hot_wt[
                        pos_index, :
                    ]
                else:  #Or di-nucleotide-shuffle
                    
                    #Get a list of shuffled dinucleotides (shift sequence by 1 every other sample)
                    if sample_ix % 2 == 0:
                        shuffled_pos_index = [
                            [pos_index[pos_j], pos_index[pos_j + 1]]
                            if pos_j + 1 < pos_index.shape[0]
                            else [pos_index[pos_j]]
                            for pos_j in range(0, pos_index.shape[0], 2)
                        ]
                    else:
                        pos_index_rev = np.copy(pos_index)[::-1]
                        shuffled_pos_index = [
                            [pos_index_rev[pos_j], pos_index_rev[pos_j + 1]]
                            if pos_j + 1 < pos_index_rev.shape[0]
                            else [pos_index_rev[pos_j]]
                            for pos_j in range(0, pos_index_rev.shape[0], 2)
                        ]

                    #Shuffle list of dinucleotide indices
                    shuffled_shuffle_index = np.arange(
                        len(shuffled_pos_index), dtype="int32"
                    )
                    np.random.shuffle(shuffled_shuffle_index)

                    #Reconstruct new list of dinucleotides
                    shuffled_pos_index_new = []
                    for pos_tuple_i in range(len(shuffled_pos_index)):
                        shuffled_pos_index_new.extend(
                            shuffled_pos_index[shuffled_shuffle_index[pos_tuple_i]]
                        )

                    #Reconstruct sequence
                    shuffled_pos_index = np.array(shuffled_pos_index_new, dtype="int32")
                    sequence_one_hot_mut[shuffled_pos_index, :] = sequence_one_hot_wt[
                        pos_index, :
                    ]

                #Get pred
                y_mut = predict_tracks(models, sequence_one_hot_mut)[0, ...][
                    ..., track_index
                ].astype("float32")

                #Undo transforms
                    
                if untransform_old :
                    #Undo scale
                    y_mut /= track_scale

                    #Undo clip-soft
                    if clip_soft is not None:
                        y_mut_unclipped = (y_mut - clip_soft) ** 2 + clip_soft
                        unclip_mask_mut = y_mut > clip_soft

                        y_mut[unclip_mask_mut] = y_mut_unclipped[unclip_mask_mut]

                    #Undo sqrt
                    y_mut = y_mut ** (1. / track_transform)
                else :
                    #Undo clip-soft
                    if clip_soft is not None :
                        y_mut_unclipped = (y_mut - clip_soft + 1)**2 + clip_soft - 1
                        unclip_mask_mut = (y_mut > clip_soft)

                        y_mut[unclip_mask_mut] = y_mut_unclipped[unclip_mask_mut]

                    #Undo sqrt
                    y_mut = (y_mut + 1)**(1. / track_transform) - 1

                    #Undo scale
                    y_mut /= track_scale

                #Aggregate over tracks (average)
                y_mut = np.mean(y_mut, axis=-1)

                #Calculate variant statistic
                score_mut = _prediction_ism_score(
                    y_mut,
                    prox_bin_start,
                    prox_bin_end,
                    dist_bin_start,
                    dist_bin_end,
                    use_mean,
                    use_ratio,
                    use_logodds,
                    prox_bin_index,
                    dist_bin_index,
                )

                pred_shuffle[example_ix, :, j, sample_ix] = score_wt - score_mut

        #Average across mutations at each position and broadcast back to nucleotides
        pred_ism[example_ix, ...] = (
            np.tile(
                np.mean(pred_shuffle[example_ix, ...], axis=-1)[..., None], (1, 1, 4)
            )
            * sequence_one_hots[example_ix][None, ...]
        )

    #Average across model replicates
    pred_ism = np.mean(pred_ism, axis=1)
    pred_ism = [
        pred_ism[example_ix, ...] for example_ix in range(len(sequence_one_hots))
    ]

    return pred_ism

#Function to visualize attribution scores as a sequence logo
def plot_seq_scores(
    importance_scores,
    figsize=(16, 2),
    plot_y_ticks=True,
    y_min=None,
    y_max=None,
    save_figs=False,
    fig_name="default",
):

    #Transpose score matrix
    importance_scores = importance_scores.T

    fig = plt.figure(figsize=figsize)

    ref_seq = ""
    #Loop over one-hot pattern and decode sequence
    for j in range(importance_scores.shape[1]):
        argmax_nt = np.argmax(np.abs(importance_scores[:, j]))

        #Decode the corresponding nucleotide that was set to 'high'
        if argmax_nt == 0:
            ref_seq += "A"
        elif argmax_nt == 1:
            ref_seq += "C"
        elif argmax_nt == 2:
            ref_seq += "G"
        elif argmax_nt == 3:
            ref_seq += "T"

    ax = plt.gca()

    #Loop over positions in the sequence and plot a DNA letter
    for i in range(0, len(ref_seq)):
        mutability_score = np.sum(importance_scores[:, i])
        color = None
        dna_letter_at(ref_seq[i], i + 0.5, 0, mutability_score, ax, color=color)

    plt.sca(ax)
    plt.xticks([], [])
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.xlim((0, len(ref_seq)))

    # plt.axis('off')

    #Remove y ticks by default
    if plot_y_ticks:
        plt.yticks(fontsize=12)
    else:
        plt.yticks([], [])

    #Set logo height
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(y_min)
    else:
        plt.ylim(
            np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores)),
            np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores)),
        )

    #Plot bottom line in the logo
    plt.axhline(y=0.0, color="black", linestyle="-", linewidth=1)

    # for axis in fig.axes :
    #    axis.get_xaxis().set_visible(False)
    #    axis.get_yaxis().set_visible(False)

    plt.tight_layout()

    #Optionally save figure
    if save_figs:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
        plt.savefig(fig_name + ".eps")

    plt.show()

#Function to visualize a pair of sequence logos with matched scales
def visualize_input_gradient_pair(
    grad_wt, grad_mut, plot_start=0, plot_end=100, save_figs=False, fig_name=""
):

    #Slice out sequence logo subplot
    scores_wt = grad_wt[plot_start:plot_end, :]
    scores_mut = grad_mut[plot_start:plot_end, :]

    #Calculate min/max range
    y_min = min(np.min(scores_wt), np.min(scores_mut))
    y_max = max(np.max(scores_wt), np.max(scores_mut))

    #Calculate absolute-valued max
    y_max_abs = max(np.abs(y_min), np.abs(y_max))

    #Add symmetric amount of padding to logos
    y_min = y_min - 0.05 * y_max_abs
    y_max = y_max + 0.05 * y_max_abs

    #Plot ref logo
    
    print("--- WT ---")
    plot_seq_scores(
        scores_wt,
        y_min=y_min,
        y_max=y_max,
        figsize=(8, 1),
        plot_y_ticks=False,
        save_figs=save_figs,
        fig_name=fig_name + "_wt",
    )

    #Plot alt logo
    
    print("--- Mut ---")
    plot_seq_scores(
        scores_mut,
        y_min=y_min,
        y_max=y_max,
        figsize=(8, 1),
        plot_y_ticks=False,
        save_figs=save_figs,
        fig_name=fig_name + "_mut",
    )

#Function to visualize coverage tracks and gene annotations
def plot_coverage_tracks(
    y_1_in,
    track_indices,
    track_names,
    track_colors,
    track_labels,
    track_scale,
    track_transform,
    clip_soft,
    start,
    y_2_in=None,
    log_scale=False,
    plot_pair=True,
    pair_alpha=0.5,
    pair_order=[0, 1],
    plot_start_rel=512,
    plot_end_rel=524288-512,
    normalize_start_rel=512,
    normalize_end_rel=524288-512,
    normalize_counts=False,
    highlight_pos_rel=None,
    highlight_covr_poses_rel=None,
    covr_orientation='before',
    covr_agg='mean',
    covr_width=4,
    bin_size=32,
    pad=16,
    same_scale=True,
    save_figs=False,
    save_suffix='default',
    fig_size=(12, 2),
    gene_slice=None,
    gene_slices=None,
    isoform_slices=None,
    gene_strand=None,
    chrom=None,
    search_gene=None,
    gene_strands=None,
    apa_df_gene_utr=None,
    apa_df_gene_intron=None,
    tss_df_gene=None,
    only_count_within_range=True,
    plot_other_genes=False,
    plot_other_gene_strands=False,
    plot_isoforms=False,
    plot_isoform_strands=False,
    max_isoforms=5,
    isoform_height_frac=0.,
    plot_strands=True,
    gene_color='black',
    isoform_color='black',
    other_gene_color='black',
    plot_as_bars=False,
    annotate_utr_apa=False,
    annotate_intron_apa=False,
    annotate_tss=False,
    untransform_old=False
) :
    
    #Calculate plot start and end bin positions
    plot_start = start + plot_start_rel
    plot_end = start + plot_end_rel
    
    plot_start_bin = plot_start_rel // bin_size - pad
    plot_end_bin = plot_end_rel // bin_size - pad
    
    #Calculate coverage normalization start and end bin positions
    normalize_start = start + normalize_start_rel
    normalize_end = start + normalize_end_rel

    normalize_start_bin = normalize_start_rel // bin_size - pad
    normalize_end_bin = normalize_end_rel // bin_size - pad
    
    #Calculate highlight coverage bin for optional annotation
    highlight_bin = None
    if highlight_pos_rel is not None :
        highlight_bin = highlight_pos_rel // bin_size - pad
    
    #Calculate highlight coverage bins for coverage ratio annotations
    highlight_covr_bins_rel = None
    if highlight_covr_poses_rel is not None :
        highlight_covr_bins_rel = [
            highlight_covr_poses_rel[0] // bin_size - pad,
            highlight_covr_poses_rel[1] // bin_size - pad,
        ]
    
    #Get gene exons
    gene_exons = []

    gene_exon = []
    for exon_ix in gene_slice.tolist() :
        if len(gene_exon) == 0 or gene_exon[-1] == exon_ix - 1 :
            gene_exon.append(exon_ix)
        else :
            gene_exons.append(gene_exon)
            gene_exon = [exon_ix]

    if len(gene_exon) > 0 :
        gene_exons.append(gene_exon)
    
    #Get exons from other genes
    other_exons = []
    for other_ix in range(len(gene_slices)) :
        other_gene_exons = []

        other_gene_exon = []
        for exon_ix in gene_slices[other_ix].tolist() :
            if len(other_gene_exon) == 0 or other_gene_exon[-1] == exon_ix - 1 :
                other_gene_exon.append(exon_ix)
            else :
                other_gene_exons.append(other_gene_exon)
                other_gene_exon = [exon_ix]

        if len(other_gene_exon) > 0 :
            other_gene_exons.append(other_gene_exon)
        
        other_exons.append(other_gene_exons)
    
    #Get isoform exons
    isoform_exons = []
    for other_ix in range(min(len(isoform_slices), max_isoforms)) :
        other_isoform_exons = []

        other_isoform_exon = []
        for exon_ix in isoform_slices[other_ix].tolist() :
            if len(other_isoform_exon) == 0 or other_isoform_exon[-1] == exon_ix - 1 :
                other_isoform_exon.append(exon_ix)
            else :
                other_isoform_exons.append(other_isoform_exon)
                other_isoform_exon = [exon_ix]

        if len(other_isoform_exon) > 0 :
            other_isoform_exons.append(other_isoform_exon)
        
        isoform_exons.append(other_isoform_exons)
    
    if y_2_in is None :
        y_2_in = np.zeros(y_1_in.shape, dtype='float32')
    
    #Copy coverage tensors
    y_1 = np.array(np.copy(y_1_in), dtype=np.float32)
    y_2 = np.array(np.copy(y_2_in), dtype=np.float32)
    
    #Broadcast data transformation parameters
    track_scales = None
    clip_softs = None
    track_transforms = None

    if not isinstance(track_scale, np.ndarray) :
        track_scales = np.array([track_scale] if not isinstance(track_scale, list) else track_scale, dtype='float32')
    else :
        track_scales = track_scale
    
    if not isinstance(clip_soft, np.ndarray) :
        clip_softs = np.array([clip_soft] if not isinstance(clip_soft, list) else clip_soft, dtype='float32')
    else :
        clip_softs = clip_soft
    
    if not isinstance(track_transform, np.ndarray) :
        track_transforms = np.array([track_transform] if not isinstance(track_transform, list) else track_transform, dtype='float32')
    else :
        track_transforms = track_transform
    
    track_scales = track_scales[None, None, None, :]
    clip_softs = clip_softs[None, None, None, :]
    track_transforms = track_transforms[None, None, None, :]

    #Undo transformations
    
    if untransform_old :
        
        #Undo scale
        y_1 /= track_scales
        y_2 /= track_scales

        #Undo clip-soft
        if clip_soft is not None :
            y_1_unclipped = (y_1 - clip_softs)**2 + clip_softs
            y_2_unclipped = (y_2 - clip_softs)**2 + clip_softs

            unclip_mask_1 = (y_1 > clip_softs)
            unclip_mask_2 = (y_2 > clip_softs)

            y_1[unclip_mask_1] = y_1_unclipped[unclip_mask_1]
            y_2[unclip_mask_2] = y_2_unclipped[unclip_mask_2]

        #Undo sqrt
        y_1 = y_1**(1. / track_transforms)
        y_2 = y_2**(1. / track_transforms)
    else :
        
        #Undo clip-soft
        if clip_soft is not None :
            y_1_unclipped = (y_1 - clip_softs + 1)**2 + clip_softs - 1
            y_2_unclipped = (y_2 - clip_softs + 1)**2 + clip_softs - 1

            unclip_mask_1 = (y_1 > clip_softs)
            unclip_mask_2 = (y_2 > clip_softs)

            y_1[unclip_mask_1] = y_1_unclipped[unclip_mask_1]
            y_2[unclip_mask_2] = y_2_unclipped[unclip_mask_2]

        #Undo sqrt
        y_1 = (y_1 + 1)**(1. / track_transforms) - 1
        y_2 = (y_2 + 1)**(1. / track_transforms) - 1

        #Undo scale
        y_1 /= track_scales
        y_2 /= track_scales
    
    #Pool replicate tracks
    y_1_pooled = []
    y_2_pooled = []
    for track_index in track_indices :
        y_1_pooled.append(np.mean(y_1[..., track_index], axis=(0, 1, 3))[:, None])
        y_2_pooled.append(np.mean(y_2[..., track_index], axis=(0, 1, 3))[:, None])
    
    y_1 = np.concatenate(y_1_pooled, axis=-1)
    y_2 = np.concatenate(y_2_pooled, axis=-1)
    
    #Optionally normalize coverage track pair counts
    if normalize_counts :
        c_1 = np.sum(y_1[normalize_start_bin:normalize_end_bin, :], axis=0)[None, :]
        c_2 = np.sum(y_2[normalize_start_bin:normalize_end_bin, :], axis=0)[None, :]

        #Normalize to densities
        y_1 /= c_1
        y_2 /= c_2

        #Bring back to count space (same reference)
        y_1 *= c_1
        y_2 *= c_1
    
    #Calculate globally largest value among track pair
    max_y = 0.
    if same_scale :
        if not log_scale :
            max_y = np.max(y_1[plot_start_bin:plot_end_bin, :])
            if plot_pair :
                max_y = max(np.max(y_1[plot_start_bin:plot_end_bin, :]), np.max(y_2[plot_start_bin:plot_end_bin, :]))
        else:
            max_y = np.max(np.log2(y_1[plot_start_bin:plot_end_bin, :] + 1.))
            if plot_pair :
                max_y = max(np.log2(y_1[plot_start_bin:plot_end_bin, :] + 1.), np.log2(y_2[plot_start_bin:plot_end_bin, :] + 1.))

    #Plot track densities as vertical-layout subplots
    f, ax = plt.subplots(len(track_labels), 1, figsize=(fig_size[0], fig_size[1] * len(track_labels)), dpi=600)
    if len(track_labels) == 1 :
        ax = [ax]
    
    #Loop over tracks
    for track_i, [track_name, track_color, track_label] in enumerate(zip(track_names, track_colors, track_labels)) :

        #Get coverage tracks for current target index
        y_1_i = y_1[..., track_i]
        y_2_i = y_2[..., track_i]

        #Aggregate coverage across target gene
        sum_1_i = 0.
        sum_2_i = 0.
        if gene_slice is not None :
            if not only_count_within_range :
                sum_1_i = np.sum(y_1_i[gene_slice])
                sum_2_i = np.sum(y_2_i[gene_slice])
            else :
                sum_1_i = np.sum(y_1_i[gene_slice[(gene_slice >= plot_start_bin) & (gene_slice < plot_end_bin)]])
                sum_2_i = np.sum(y_2_i[gene_slice[(gene_slice >= plot_start_bin) & (gene_slice < plot_end_bin)]])
        
        #Save a copy of the raw coverage tracks
        y_1_i_raw = np.copy(y_1_i)
        y_2_i_raw = np.copy(y_2_i)
        
        #Slice out position interval
        y_1_i = y_1_i[plot_start_bin:plot_end_bin]
        y_2_i = y_2_i[plot_start_bin:plot_end_bin]
        
        #Optional log+1 transform
        if log_scale :
            y_1_i = np.log2(y_1_i + 1.)
            y_2_i = np.log2(y_2_i + 1.)
        
        #Calculate max values per track
        max_1_i = np.max(y_1_i)
        max_2_i = np.max(y_2_i)

        if plot_pair :
            max_y_i = max(max_1_i, max_2_i)
        else :
            max_y_i = max_1_i
        
        if same_scale :
            max_y_i = max_y

        plt.sca(ax[track_i])
        
        legend_handles = []

        #Plot tracks as colored curve areas
        if not plot_as_bars :
            h1 = ax[track_i].fill_between(
                np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
                y_1_i,
                color=track_color[0],
                alpha=pair_alpha,
                label=track_label[0] + " - " + track_name,
                zorder=pair_order[0],
                rasterized=True
            )
            legend_handles.append(h1)

            if plot_pair :
                h2 = ax[track_i].fill_between(
                    np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
                    y_2_i,
                    color=track_color[1],
                    alpha=pair_alpha,
                    label=track_label[1] + " - " + track_name,
                    zorder=pair_order[1],
                    rasterized=True
                )
                legend_handles.append(h2)
        else : #Or plot tracks as bars (non-rasterized)
            plt.bar(
                np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
                y_1_i,
                width=1,
                color=track_color[0],
                alpha=pair_alpha,
                label=track_label[0] + " - " + track_name,
                zorder=pair_order[0]
            )

            if plot_pair :
                plt.bar(
                    np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
                    y_2_i,
                    width=1,
                    color=track_color[1],
                    alpha=pair_alpha,
                    label=track_label[1] + " - " + track_name,
                    zorder=pair_order[1]
                )

        #Annotate gene exons
        for gene_exon_i, gene_exon in enumerate(gene_exons) :
            exon_start_bin = gene_exon[0] - 0.5
            exon_end_bin = gene_exon[-1] + 0.5
            
            #Plot shaded blue area around exon coverage peaks
            if (gene_exon[-1] >= plot_start_bin and gene_exon[0] < plot_end_bin) :
                ax[track_i].fill_between([exon_start_bin, exon_end_bin], max_y_i * 0.9995, color='deepskyblue', alpha=0.1, zorder=3)
        
        plt.sca(ax[track_i])
        
        #Plot annotation graphics for the current gene (union of exons)
        if gene_slice is not None :
            #Plot entire gene span as line
            plt.plot([gene_exons[0][0], gene_exons[-1][-1]], [-0.075 * max_y_i, -0.075 * max_y_i], zorder=5, color=gene_color, linewidth=0.5, linestyle='--')
            
            #Loop over exon starts and ends
            for gene_exon_i, gene_exon in enumerate(gene_exons) :
                exon_start_bin = gene_exon[0] - 0.5
                exon_end_bin = gene_exon[-1] + 0.5#1.5
                
                #Plot exon as rectangle within gene span
                rect = patches.Rectangle((exon_start_bin, -0.10 * max_y_i), (exon_end_bin - exon_start_bin), 0.05 * max_y_i, linewidth=0.5, edgecolor=gene_color, facecolor=gene_color, zorder=6)
                ax[track_i].add_patch(rect)
                
                #Optionally plot gene strandedness as arrows along introns in gene span
                if plot_strands and gene_exon_i < len(gene_exons) - 1 :
                    next_exon_start_bin = gene_exons[gene_exon_i+1][0] - 0.5
                    intron_mid = (exon_end_bin + next_exon_start_bin) / 2.
                    
                    arrow_len = 0.004 * (plot_end_bin - plot_start_bin)
                    intron_len = next_exon_start_bin - exon_end_bin
                    
                    #Only plot if the arrow fits neatly within the intron
                    if intron_len >= 2 * arrow_len :
                        strand_sign = -1. if gene_strand == '-' else 1.
                        strand_arrow = patches.FancyArrow(intron_mid - (arrow_len/2.) * strand_sign, -0.075 * max_y_i, arrow_len * strand_sign, 0., length_includes_head=True, width=0., head_width=0.04 * max_y_i, head_length=arrow_len, zorder=7, color=gene_color)
                        ax[track_i].add_patch(strand_arrow)
        
        #Optionally highlight bin of interest
        if highlight_bin is not None :
            l1 = plt.plot([highlight_bin, highlight_bin], [0., max_y * 0.9995], color='black', linewidth=0.5, linestyle='--', alpha=0.5, zorder=10, label='Highlight')
            #legend_handles.append(l1[0])
        
        #Optionally annotate pA sites (3' UTR)
        if annotate_utr_apa :
            site_poses = apa_df_gene_utr.query("chrom == '" + chrom + "' and position_hg38 >= " + str(plot_start) + " and position_hg38 < " + str(plot_end))['position_hg38'].values.tolist()
            
            #Loop over pA sites
            for site_ix, site_pos in enumerate(site_poses) :
                site_bin = int((site_pos - start) // bin_size) - pad
                #site_bin = int(np.round((site_pos - start) / bin_size)) - pad
                
                l1 = plt.plot([site_bin, site_bin], [0., max_y_i * 0.9995], color='maroon', linewidth=0.5, alpha=0.5, linestyle='--', zorder=10, label='PAS')
                #if site_ix == 0 :
                #    legend_handles.append(l1[0])
        
        #Optionally annotate intronic pA sites
        if annotate_intron_apa :
            site_poses = apa_df_gene_intron.query("chrom == '" + chrom + "' and position_hg38 >= " + str(plot_start) + " and position_hg38 < " + str(plot_end))['position_hg38'].values.tolist()
            
            #Loop over intronic pA sites
            for site_ix, site_pos in enumerate(site_poses) :
                site_bin = int((site_pos - start) // bin_size) - pad
                #site_bin = int(np.round((site_pos - start) / bin_size)) - pad
                
                plt.plot([site_bin, site_bin], [0., max_y_i * 0.9995], color='maroon', linewidth=0.5, alpha=0.5, linestyle='--', zorder=10)
        
        #Optionally annotate TSS positions
        if annotate_tss :
            site_poses = tss_df_gene.query("chrom == '" + chrom + "' and position_hg38 >= " + str(plot_start) + " and position_hg38 < " + str(plot_end))['position_hg38'].values.tolist()
            
            #Loop over TSS positions
            for site_ix, site_pos in enumerate(site_poses) :
                site_bin = int((site_pos - start) // bin_size) - pad
                #site_bin = int(np.round((site_pos - start) / bin_size)) - pad
                
                l1 = plt.plot([site_bin, site_bin], [0., max_y_i * 0.9995], color='darkgreen', linewidth=0.5, alpha=0.5, linestyle='--', zorder=10, label='TSS')
                #if site_ix == 0 :
                #    legend_handles.append(l1[0])
        
        #Optionally annotate regions used to estimate coverage ratios
        y_1_site_1_cov = 0.
        y_2_site_1_cov = 0.
        y_1_site_2_cov = 0.
        y_2_site_2_cov = 0.
        if highlight_covr_bins_rel is not None :
            
            site_1_bin = highlight_covr_bins_rel[0]
            site_2_bin = highlight_covr_bins_rel[1]
            
            bin_1_start = None
            bin_1_end = None
            bin_2_start = None
            bin_2_end = None
            if covr_orientation == 'before' :
                if gene_strand == '+' :
                    bin_1_end = site_1_bin + 1
                    bin_1_start = bin_1_end - covr_width
                    bin_2_end = site_2_bin + 1
                    bin_2_start = bin_2_end - covr_width
                else :
                    bin_1_start = site_1_bin
                    bin_1_end = bin_1_start + covr_width
                    bin_2_start = site_2_bin
                    bin_2_end = bin_2_start + covr_width
            else :
                if gene_strand == '+' :
                    bin_1_start = site_1_bin
                    bin_1_end = bin_1_start + covr_width
                    bin_2_start = site_2_bin
                    bin_2_end = bin_2_start + covr_width
                else :
                    bin_1_end = site_1_bin + 1
                    bin_1_start = bin_1_end - covr_width
                    bin_2_end = site_2_bin + 1
                    bin_2_start = bin_2_end - covr_width
            
            if covr_agg == 'mean' :
                y_1_site_1_cov = np.mean(y_1_i_raw[bin_1_start:bin_1_end])
                y_1_site_2_cov = np.mean(y_1_i_raw[bin_2_start:bin_2_end])
                y_2_site_1_cov = np.mean(y_2_i_raw[bin_1_start:bin_1_end])
                y_2_site_2_cov = np.mean(y_2_i_raw[bin_2_start:bin_2_end])
            elif covr_agg == 'max' :
                y_1_site_1_cov = np.max(y_1_i_raw[bin_1_start:bin_1_end])
                y_1_site_2_cov = np.max(y_1_i_raw[bin_2_start:bin_2_end])
                y_2_site_1_cov = np.max(y_2_i_raw[bin_1_start:bin_1_end])
                y_2_site_2_cov = np.max(y_2_i_raw[bin_2_start:bin_2_end])
            
            plt.plot([bin_1_start-0.5, bin_1_end-1+0.5], [0.99 * max_y_i, 0.99 * max_y_i], linewidth=0.5, linestyle='-', color='black', zorder=11)
            plt.plot([bin_1_start-0.5, bin_1_start-0.5], [0.95 * max_y_i, 0.99 * max_y_i], linewidth=0.5, linestyle='-', color='black', zorder=11)
            plt.plot([bin_1_end-1+0.5, bin_1_end-1+0.5], [0.95 * max_y_i, 0.99 * max_y_i], linewidth=0.5, linestyle='-', color='black', zorder=11)
            
            plt.plot([bin_2_start-0.5, bin_2_end-1+0.5], [0.99 * max_y_i, 0.99 * max_y_i], linewidth=0.5, linestyle='-', color='black', zorder=11)
            plt.plot([bin_2_start-0.5, bin_2_start-0.5], [0.95 * max_y_i, 0.99 * max_y_i], linewidth=0.5, linestyle='-', color='black', zorder=11)
            plt.plot([bin_2_end-1+0.5, bin_2_end-1+0.5], [0.95 * max_y_i, 0.99 * max_y_i], linewidth=0.5, linestyle='-', color='black', zorder=11)
            
            rect_1 = patches.Rectangle((bin_1_start-0.5, 0.975 * max_y_i), (bin_1_end - bin_1_start), (0.99 - 0.975) * max_y_i, linewidth=0., facecolor='lightcoral', alpha=0.35, zorder=11)
            rect_2 = patches.Rectangle((bin_2_start-0.5, 0.975 * max_y_i), (bin_2_end - bin_2_start), (0.99 - 0.975) * max_y_i, linewidth=0., facecolor='lightcoral', alpha=0.35, zorder=11)
            ax[track_i].add_patch(rect_1)
            ax[track_i].add_patch(rect_2)
        
        #Optionally plot the union of exons of other genes
        if plot_other_genes :
            
            #Loop over other genes
            for other_ix in range(len(other_exons)) :
                plt.plot([other_exons[other_ix][0][0], other_exons[other_ix][-1][-1]], [(-0.075 - 0.10 - isoform_height_frac) * max_y_i, (-0.075 - 0.10 - isoform_height_frac) * max_y_i], zorder=5, color=other_gene_color, linewidth=0.5, linestyle='--')

                #Loop over the exons of the current other gene
                for gene_exon_i, gene_exon in enumerate(other_exons[other_ix]) :
                    exon_start_bin = gene_exon[0] - 0.5
                    exon_end_bin = gene_exon[-1] + 0.5

                    #Plot exon graphic
                    rect = patches.Rectangle((exon_start_bin, (-0.10 - 0.10 - isoform_height_frac) * max_y_i), (exon_end_bin - exon_start_bin), 0.05 * max_y_i, linewidth=0.5, edgecolor=other_gene_color, facecolor=other_gene_color, zorder=6)
                    ax[track_i].add_patch(rect)
                    
                    #Plot gene strandedness of other genes
                    if plot_other_gene_strands and gene_exon_i < len(other_exons[other_ix]) - 1 :
                        next_exon_start_bin = other_exons[other_ix][gene_exon_i+1][0] - 0.5
                        intron_mid = (exon_end_bin + next_exon_start_bin) / 2.

                        arrow_len = 0.004 * (plot_end_bin - plot_start_bin)
                        intron_len = next_exon_start_bin - exon_end_bin
                        
                        #Plot arrow only if intron is wide enough
                        if intron_len >= 2 * arrow_len :
                            strand_sign = -1. if gene_strands[other_ix] == '-' else 1.
                            strand_arrow = patches.FancyArrow(intron_mid - (arrow_len/2.) * strand_sign, (-0.075 - 0.10 - isoform_height_frac) * max_y_i, arrow_len * strand_sign, 0., length_includes_head=True, width=0., head_width=0.04 * max_y_i, head_length=arrow_len, zorder=7, color=other_gene_color)
                            ax[track_i].add_patch(strand_arrow)
        
        #Annotate a selection of isoforms of the target gene
        if plot_isoforms :
            
            #Loop over isoforms
            for isoform_ix in range(len(isoform_exons)) :
                isoform_offset = (isoform_ix + 1) * 0.10
                next_isoform_offset = (isoform_ix + 2) * 0.10
                
                #Plot only if isoform will fit in alloted relative area within subplot
                if isoform_ix == len(isoform_exons) - 1 or next_isoform_offset <= isoform_height_frac :
                    plt.plot([isoform_exons[isoform_ix][0][0], isoform_exons[isoform_ix][-1][-1]], [(-0.075 - isoform_offset) * max_y_i, (-0.075 - isoform_offset) * max_y_i], zorder=5, color=isoform_color, linewidth=0.5, linestyle='--')

                    #Loop over the exons of the current isoform
                    for gene_exon_i, gene_exon in enumerate(isoform_exons[isoform_ix]) :
                        exon_start_bin = gene_exon[0] - 0.5
                        exon_end_bin = gene_exon[-1] + 0.5

                        rect = patches.Rectangle((exon_start_bin, (-0.10 - isoform_offset) * max_y_i), (exon_end_bin - exon_start_bin), 0.05 * max_y_i, linewidth=0.5, edgecolor=isoform_color, facecolor=isoform_color, zorder=6)
                        ax[track_i].add_patch(rect)
                    
                        #Plot gene strandedness along isoform
                        if plot_isoform_strands and gene_exon_i < len(isoform_exons[isoform_ix]) - 1 :
                            next_exon_start_bin = isoform_exons[isoform_ix][gene_exon_i+1][0] - 0.5
                            intron_mid = (exon_end_bin + next_exon_start_bin) / 2.

                            arrow_len = 0.004 * (plot_end_bin - plot_start_bin)
                            intron_len = next_exon_start_bin - exon_end_bin

                            #Plot arrow only if intron is wide enough
                            if intron_len >= 2 * arrow_len :
                                strand_sign = -1. if gene_strand == '-' else 1.
                                strand_arrow = patches.FancyArrow(intron_mid - (arrow_len/2.) * strand_sign, (-0.075 - isoform_offset) * max_y_i, arrow_len * strand_sign, 0., length_includes_head=True, width=0., head_width=0.04 * max_y_i, head_length=arrow_len, zorder=7, color=isoform_color)
                                ax[track_i].add_patch(strand_arrow)
                
                #Plot text if there are too many isoforms to show
                else :
                    missing_isoforms = len(isoform_slices) - isoform_ix
                    plt.text((plot_start_bin + (plot_end_bin-1)) / 2., (-0.075 - isoform_offset) * max_y_i, "(+" + str(missing_isoforms) + " not shown...)", horizontalalignment='center', verticalalignment='center', fontsize=6, zorder=10)
                    break
        
        plt.axvline(x=plot_start_bin, linewidth=1, linestyle='-', color='black')
        
        #Apply subplot limits
        plt.xlim(plot_start_bin, plot_end_bin-1)
        if gene_slice is not None :
            if plot_other_genes :
                plt.ylim((-0.25 - isoform_height_frac) * max_y_i, max_y_i)
            else :
                plt.ylim((-0.15 - isoform_height_frac) * max_y_i, max_y_i)
        else :
            plt.ylim(0., max_y_i)

        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.axis('off')

        #Annotate chromosome and coordinates plotted
        if track_i == len(track_labels) - 1 :
            text_str = chrom + ":" + str(plot_start) + "-" + str(plot_end) + " (" + str(int(plot_end-plot_start)) + "bp) - " + "'" + search_gene + "' (" + gene_strand + ")"
            plt.text(0.0, -0.14 / float(fig_size[1]), text_str, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, zorder=6)
        
        #Annotate metrics derived from the coverage tracks
        if gene_slice is not None :
            tr_label_0_str = ' (' + track_label[0] + ')' if plot_pair else ''
            tr_label_1_str = ' (' + track_label[1] + ')' if plot_pair else ''
            
            #Max coverage
            y_max_str = 'Max' + tr_label_0_str + ' = ' + str(round(max_1_i if not log_scale else 2**max_1_i - 1, 2))
            if plot_pair :
                y_max_str += ',' + tr_label_1_str + ' = ' + str(round(max_2_i if not log_scale else 2**max_2_i - 1, 2))
            
            #Sum of coverage
            y_sum_str = 'Sum' + tr_label_0_str + ' = ' + str(round(sum_1_i, 2))
            if plot_pair :
                y_sum_str += ',' + tr_label_1_str + ' = ' + str(round(sum_2_i, 2))
            
            plt.text(0.005, 0.94, y_max_str, fontname='monospace', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=6, zorder=6)
            plt.text(0.005, 0.82, y_sum_str, fontname='monospace', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=6, zorder=6)
            
            #Annotate log fold change (if plotting a pair of coverage tracks)
            if plot_pair and track_label[0].lower() in ['ref', 'wt'] and track_label[1].lower() in ['alt', 'var', 'mut'] :
                log_ratio_str = 'Log ratio (' + track_label[1] + ' / ' + track_label[0] + ') = ' + str(round(np.log2(sum_2_i / sum_1_i), 3))
                plt.text(0.005, 0.70, log_ratio_str, fontname='monospace', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=6, zorder=6)
        
        #Optionally annotate coverage ratio metrics computed from the tracks
        if highlight_covr_bins_rel is not None :
            
            covr_1_i = 0.
            covr_2_i = 0.
            if gene_strand == '-' :
                covr_1_i = (y_1_site_1_cov + 1e-6) / (y_1_site_2_cov + 1e-6)
                covr_2_i = (y_2_site_1_cov + 1e-6) / (y_2_site_2_cov + 1e-6)
            else :
                covr_1_i = (y_1_site_2_cov + 1e-6) / (y_1_site_1_cov + 1e-6)
                covr_2_i = (y_2_site_2_cov + 1e-6) / (y_2_site_1_cov + 1e-6)
            
            #Coverage ratio
            covr_str = 'COVR' + tr_label_0_str + ' = ' + str(round(covr_1_i, 3))
            if plot_pair :
                covr_str += ',' + tr_label_1_str + ' = ' + str(round(covr_2_i, 3))

            plt.text(0.005, 0.58, covr_str, fontname='monospace', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=6, zorder=6)
            
            #Annotate log fold change (if plotting a pair of coverage tracks)
            if plot_pair and track_label[0].lower() in ['ref', 'wt'] and track_label[1].lower() in ['alt', 'var', 'mut'] :
                log_ratio_str = 'Log COVR ratio (' + track_label[1] + ' / ' + track_label[0] + ') = ' + str(round(np.log2(covr_2_i / covr_1_i), 3))
                plt.text(0.005, 0.46, log_ratio_str, fontname='monospace', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=6, zorder=6)
        
        plt.legend(handles=legend_handles, loc='upper right', fontsize=6)

    plt.tight_layout()
    
    #Optionally save figure
    if save_figs :
        plt.savefig("borzoi" + save_suffix + ".png", dpi=300, transparent=False)
        plt.savefig("borzoi" + save_suffix + ".pdf")

    plt.show()


#Function to visualize coverage tracks
def plot_coverage_track_pair_bins(
    y_wt,
    y_mut,
    chrom,
    start,
    center_pos,
    poses,
    track_indices,
    track_names,
    track_scales,
    track_transforms,
    clip_softs,
    log_scale=False,
    sqrt_scale=False,
    plot_mut=True,
    plot_window=4096,
    normalize_window=4096,
    bin_size=32,
    pad=16,
    normalize_counts=False,
    save_figs=False,
    save_suffix="default",
    gene_slice=None,
    anno_df=None,
    untransform_old=False
):

    #Calculate plot start and end bin positions
    plot_start = center_pos - plot_window // 2
    plot_end = center_pos + plot_window // 2

    plot_start_bin = (plot_start - start) // bin_size - pad
    plot_end_bin = (plot_end - start) // bin_size - pad

    #Calculate coverage normalization start and end bin positions
    normalize_start = center_pos - normalize_window // 2
    normalize_end = center_pos + normalize_window // 2

    normalize_start_bin = (normalize_start - start) // bin_size - pad
    normalize_end_bin = (normalize_end - start) // bin_size - pad

    center_bin = (center_pos - start) // bin_size - pad
    mut_bin = (poses[0] - start) // bin_size - pad

    # Get annotation positions
    anno_poses = []
    if anno_df is not None:
        anno_poses = anno_df.query(
            "chrom == '"
            + chrom
            + "' and position_hg38 >= "
            + str(plot_start)
            + " and position_hg38 < "
            + str(plot_end)
        )["position_hg38"].values.tolist()

    # Plot each tracks
    for track_name, track_index, track_scale, track_transform, clip_soft in zip(
        track_names, track_indices, track_scales, track_transforms, clip_softs
    ):

        # Plot track densities (bins)
        y_wt_curr = np.array(np.copy(y_wt), dtype=np.float32)
        y_mut_curr = np.array(np.copy(y_mut), dtype=np.float32)

        #Undo transformations
        if untransform_old :

            #Undo scale
            y_wt_curr /= track_scale
            y_mut_curr /= track_scale

            #Undo clip-soft
            if clip_soft is not None:
                y_wt_curr_unclipped = (y_wt_curr - clip_soft) ** 2 + clip_soft
                y_mut_curr_unclipped = (y_mut_curr - clip_soft) ** 2 + clip_soft

                unclip_mask_wt = y_wt_curr > clip_soft
                unclip_mask_mut = y_mut_curr > clip_soft

                y_wt_curr[unclip_mask_wt] = y_wt_curr_unclipped[unclip_mask_wt]
                y_mut_curr[unclip_mask_mut] = y_mut_curr_unclipped[unclip_mask_mut]

            #Undo sqrt
            y_wt_curr = y_wt_curr ** (1. / track_transform)
            y_mut_curr = y_mut_curr ** (1. / track_transform)
        else :
            
            #Undo clip-soft
            if clip_soft is not None:
                y_wt_curr_unclipped = (y_wt_curr - clip_soft + 1) ** 2 + clip_soft - 1
                y_mut_curr_unclipped = (y_mut_curr - clip_soft + 1) ** 2 + clip_soft - 1

                unclip_mask_wt = y_wt_curr > clip_soft
                unclip_mask_mut = y_mut_curr > clip_soft

                y_wt_curr[unclip_mask_wt] = y_wt_curr_unclipped[unclip_mask_wt]
                y_mut_curr[unclip_mask_mut] = y_mut_curr_unclipped[unclip_mask_mut]

            #Undo sqrt
            y_wt_curr = (y_wt_curr + 1) ** (1. / track_transform) - 1
            y_mut_curr = (y_mut_curr + 1) ** (1. / track_transform) - 1
            
            #Undo scale
            y_wt_curr /= track_scale
            y_mut_curr /= track_scale

        #Average across replicate tracks
        y_wt_curr = np.mean(y_wt_curr[..., track_index], axis=(0, 1, 3))
        y_mut_curr = np.mean(y_mut_curr[..., track_index], axis=(0, 1, 3))

        #Normalize reference/alternate coverage track counts
        if normalize_counts:
            wt_count = np.sum(y_wt_curr[normalize_start_bin:normalize_end_bin])
            mut_count = np.sum(y_mut_curr[normalize_start_bin:normalize_end_bin])

            # Normalize to densities
            y_wt_curr /= wt_count
            y_mut_curr /= mut_count

            # Bring back to count space (wt reference)
            y_wt_curr *= wt_count
            y_mut_curr *= wt_count

        #Print aggregated exon coverage for target gene
        if gene_slice is not None:
            sum_wt = np.sum(y_wt_curr[gene_slice])
            sum_mut = np.sum(y_mut_curr[gene_slice])

            print(" - sum_wt = " + str(round(sum_wt, 4)))
            print(" - sum_mut = " + str(round(sum_mut, 4)))

        y_wt_curr = y_wt_curr[plot_start_bin:plot_end_bin]
        y_mut_curr = y_mut_curr[plot_start_bin:plot_end_bin]

        #Apply log+1 or sqrt+1 transform
        if log_scale:
            y_wt_curr = np.log2(y_wt_curr + 1.)
            y_mut_curr = np.log2(y_mut_curr + 1.)
        elif sqrt_scale:
            y_wt_curr = np.sqrt(y_wt_curr + 1.)
            y_mut_curr = np.sqrt(y_mut_curr + 1.)

        #Calculate global coverage peak max and print values
        max_y_wt = np.max(y_wt_curr)
        max_y_mut = np.max(y_mut_curr)

        if plot_mut:
            max_y = max(max_y_wt, max_y_mut)
        else:
            max_y = max_y_wt

        print(" - max_y_wt = " + str(round(max_y_wt, 4)))
        print(" - max_y_mut = " + str(round(max_y_mut, 4)))
        print(" -- (max_y = " + str(round(max_y, 4)) + ")")

        f = plt.figure(figsize=(12, 2))

        #Plot coverage tracks as bins
        plt.bar(
            np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
            y_wt_curr,
            width=1.0,
            color="green",
            alpha=0.5,
            label="Ref",
        )

        #Plot variant coverage tracks
        if plot_mut:
            plt.bar(
                np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
                y_mut_curr,
                width=1.0,
                color="red",
                alpha=0.5,
                label="Alt",
            )

        xtick_vals = []

        #Annotate sites from a list of positions (draw as vertical lines)
        for _, anno_pos in enumerate(anno_poses):

            anno_bin = int((anno_pos - start) // 32) - 16

            xtick_vals.append(anno_bin)

            bin_end = anno_bin + 3 - 0.5
            bin_start = bin_end - 5

            plt.axvline(
                x=anno_bin,
                color="cyan",
                linewidth=2,
                alpha=0.5,
                linestyle="-",
                zorder=-1,
            )

        #Annotate variant position
        plt.scatter(
            [mut_bin],
            [0.075 * max_y],
            color="black",
            s=125,
            marker="*",
            zorder=100,
            label="SNP",
        )

        plt.xlim(plot_start_bin, plot_end_bin - 1)

        plt.xticks([], [])
        plt.yticks([], [])

        #Annotate the plotted coordinates
        plt.xlabel(
            chrom
            + ":"
            + str(plot_start)
            + "-"
            + str(plot_end)
            + " ("
            + str(plot_window)
            + "bp window)",
            fontsize=8,
        )
        plt.ylabel("Signal", fontsize=8)

        plt.title("Track(s): " + str(track_name), fontsize=8)

        plt.legend(fontsize=8)

        plt.tight_layout()

        #Optionally save figures
        if save_figs:
            plt.savefig(
                "borzoi"
                + save_suffix
                + "_track_"
                + str(track_index[0])
                + "_to_"
                + str(track_index[-1])
                + ".png",
                dpi=300,
                transparent=False,
            )
            plt.savefig(
                "borzoi"
                + save_suffix
                + "_track_"
                + str(track_index[0])
                + "_to_"
                + str(track_index[-1])
                + ".eps"
            )

        plt.show()


# Helper functions (measured RNA-seq coverage loader)

#Function that opens coverage files and returns read and close functions
def get_coverage_reader(
    cov_files, target_length, crop_length, blacklist_bed, blacklist_pct=0.5
):

    #Open genome coverage files
    cov_opens = [CovFace(cov_file) for cov_file in cov_files]

    #Read blacklist regions
    black_chr_trees = read_blacklist(blacklist_bed)

    #Function to read coverage
    def _read_coverage(
        chrom,
        start,
        end,
        clip_soft=None,
        clip=None,
        scale=0.01,
        blacklist_pct=blacklist_pct,
        cov_opens=cov_opens,
        target_length=target_length,
        crop_length=crop_length,
        black_chr_trees=black_chr_trees,
        transform_old=False,
    ):

        n_targets = len(cov_opens)

        targets = []

        #Loop over targets
        for target_i in range(n_targets):

            #Extract sequence as BED style
            if start < 0:
                seq_cov_nt = np.concatenate(
                    [np.zeros(-start), cov_opens[target_i].read(chrom, 0, end)], axis=0
                )
            else:
                seq_cov_nt = cov_opens[target_i].read(chrom, start, end)  # start - 1

            #Extend to full length
            if seq_cov_nt.shape[0] < end - start:
                seq_cov_nt = np.concatenate(
                    [seq_cov_nt, np.zeros((end - start) - seq_cov_nt.shape[0])], axis=0
                )

            #Read coverage
            seq_cov_nt = cov_opens[target_i].read(chrom, start, end)

            #Determine baseline coverage
            if target_length >= 8:
                baseline_cov = np.percentile(seq_cov_nt, 100 * blacklist_pct)
                baseline_cov = np.nan_to_num(baseline_cov)
            else:
                baseline_cov = 0

            #Set blacklist to baseline
            if chrom in black_chr_trees:
                for black_interval in black_chr_trees[chrom][start:end]:
                    
                    #Adjust for sequence indexes
                    black_seq_start = black_interval.begin - start
                    black_seq_end = black_interval.end - start
                    black_seq_values = seq_cov_nt[black_seq_start:black_seq_end]
                    seq_cov_nt[black_seq_start:black_seq_end] = np.clip(
                        black_seq_values, -baseline_cov, baseline_cov
                    )

            #Set NaN's to baseline
            nan_mask = np.isnan(seq_cov_nt)
            seq_cov_nt[nan_mask] = baseline_cov

            #Apply original transform (from borzoi manuscript)
            if transform_old:
            
                #Sum pool
                seq_cov = (
                    seq_cov_nt.reshape(target_length, -1).sum(axis=1, dtype="float32")
                    ** 0.75
                )

                #Crop
                if crop_length > 0 :
                    seq_cov = seq_cov[crop_length:-crop_length]

                #Clip
                if clip_soft is not None:
                    clip_mask = seq_cov > clip_soft
                    seq_cov[clip_mask] = clip_soft + np.sqrt(seq_cov[clip_mask] - clip_soft)
                if clip is not None:
                    seq_cov = np.clip(seq_cov, -clip, clip)

                #Scale
                seq_cov = scale * seq_cov
            else:
                
                #Scale
                seq_cov_nt = scale * seq_cov_nt

                #Sum pool
                seq_cov = -1 + np.sqrt(
                    1 + seq_cov_nt.reshape(target_length, -1).sum(axis=1, dtype="float32")
                )
                
                #Clip
                if clip_soft is not None:
                    clip_mask = seq_cov > clip_soft
                    seq_cov[clip_mask] = clip_soft - 1 + np.sqrt(seq_cov[clip_mask] - clip_soft + 1)
                if clip is not None:
                    seq_cov = np.clip(seq_cov, -clip, clip)

            #Clip float16 min/max
            seq_cov = np.clip(
                seq_cov, np.finfo(np.float16).min, np.finfo(np.float16).max
            )

            #Append to targets
            targets.append(seq_cov.astype("float16")[:, None])

        return np.concatenate(targets, axis=-1)

    #Function to close coverage files
    def _close_coverage(cov_opens=cov_opens):
        #Loop over coverage files and close them
        for cov_open in cov_opens:
            cov_open.close()

    return _read_coverage, _close_coverage

#Function to read genome blacklist coordinates and construct interval trees
def read_blacklist(blacklist_bed, black_buffer=20):
    black_chr_trees = {}

    if blacklist_bed is not None and os.path.isfile(blacklist_bed):
        
        #Loop over blacklist
        for line in open(blacklist_bed):
            a = line.split()
            chrm = a[0]
            start = max(0, int(a[1]) - black_buffer)
            end = int(a[2]) + black_buffer

            #Initialize new interval tree for chromosome
            if chrm not in black_chr_trees:
                black_chr_trees[chrm] = intervaltree.IntervalTree()

            black_chr_trees[chrm][start:end] = True

    return black_chr_trees

#Coverage reader interface
class CovFace:
    def __init__(self, cov_file):
        self.cov_file = cov_file
        self.bigwig = False
        self.bed = False

        #Parse coverage file type and open the file
        cov_ext = os.path.splitext(self.cov_file)[1].lower()
        if cov_ext == ".gz":
            cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()

        if cov_ext in [".bed", ".narrowpeak"]:
            self.bed = True
            self.preprocess_bed()

        elif cov_ext in [".bw", ".bigwig"]:
            self.cov_open = pyBigWig.open(self.cov_file, "r")
            self.bigwig = True

        elif cov_ext in [".h5", ".hdf5", ".w5", ".wdf5"]:
            self.cov_open = h5py.File(self.cov_file, "r")

        else:
            print(
                'Cannot identify coverage file extension "%s".' % cov_ext,
                file=sys.stderr,
            )
            exit(1)

    #Function to read bed file with coordinates
    def preprocess_bed(self):
        #Read bed
        bed_df = pd.read_csv(
            self.cov_file, sep="\t", usecols=range(3), names=["chr", "start", "end"]
        )

        #Loop over chromosomes
        self.cov_open = {}
        for chrm in bed_df.chr.unique():
            bed_chr_df = bed_df[bed_df.chr == chrm]

            #Find max pos
            pos_max = bed_chr_df.end.max()

            #Initialize array
            self.cov_open[chrm] = np.zeros(pos_max, dtype="bool")

            #Set peaks
            for peak in bed_chr_df.itertuples():
                self.cov_open[peak.chr][peak.start : peak.end] = 1

    #Function to read coverage values
    def read(self, chrm, start, end):
        #Read from bigwig
        if self.bigwig:
            cov = self.cov_open.values(chrm, start, end, numpy=True).astype("float16")

        else:
            #Read from non-bigwig source
            if chrm in self.cov_open:
                cov = self.cov_open[chrm][start:end]
                pad_zeros = end - start - len(cov)
                if pad_zeros > 0:
                    cov_pad = np.zeros(pad_zeros, dtype="bool")
                    cov = np.concatenate([cov, cov_pad])
            else:
                #Error finding coordinates
                print(
                    "WARNING: %s doesn't see %s:%d-%d. Setting to all zeros."
                    % (self.cov_file, chrm, start, end),
                    file=sys.stderr,
                )
                #Return zeros
                cov = np.zeros(end - start, dtype="float16")

        return cov

    #Function to close coverage file handle
    def close(self):
        if not self.bed:
            self.cov_open.close()
