import os
import numpy as np
import pandas as pd
import pysam

from Bio import SeqIO
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gc
import subprocess
import shutil


def get_new_gene_ver(ensg, df_genes):
    """Get new gene version from dataframe."""
    df_genes_ = df_genes[df_genes['name1_trunc']==ensg.split('.')[0]]
    if len(df_genes_) > 0:
        return str(df_genes_.iloc[0]['name1'])
    else:
        return 'None'


def sort_gene_annotations(df):
    """
    Sort gene annotations dataframe by gene name ('name_full') first,
    then by transcript start position ('txStart') within each gene group.
    
    Args:
        df (pandas.DataFrame): DataFrame containing gene annotations with columns 'name_full' and 'txStart'.
        
    Returns:
        pandas.DataFrame: Sorted dataframe.
    """
    # First ensure the dataframe has the required columns
    required_columns = ['name_full', 'txStart']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Sort the dataframe by gene name first, then by transcript start position
    sorted_df = df.sort_values(by=['name_full', 'txStart'])
    
    # Reset index after sorting
    sorted_df = sorted_df.reset_index(drop=True)
        
    return sorted_df


def plot_coverage_tracks_plotly(ref_coverage, alt_coverage, variant_pos, variant_name, transcripts_toplot, 
                               xlim=None, ylim=None, fig_width=800, fig_height=None, savedir='.'):
    """Plots reference and alternative coverage tracks with variant highlight and exon annotations using Plotly.

    Args:
        ref_coverage: Array of coverage values for the reference allele.
        alt_coverage: Array of coverage values for the alternative allele.
        variant_pos: Position (index) of the variant in the coverage arrays.
        transcripts_toplot: Dictionary of transcripts to plot, with transcript name as key and list of exon coordinates as value.
        xlim: Tuple of (min, max) x-axis limits.
        ylim: Tuple of (min, max) y-axis limits.
        fig_width: Width of the figure in pixels (optional).
        fig_height: Height of the figure in pixels (optional).
        savedir: Directory to save the figure to.
    """

    if len(variant_name)>20:
        variant_name = '_'.join(variant_name.split('_')[0:2])
        
    # Set up figure
    num_transcripts = len(transcripts_toplot.keys())
    # Calculate appropriate figure height if not provided
    if fig_height is None:
        fig_height = int(0.25 * fig_width) + 80 * num_transcripts  # Base height plus room for each transcript
    
    # Set xlim if not provided
    if xlim is None:
        xlim = [0, len(ref_coverage)]
    
    # Calculate row heights - coverage plot is taller than exon plots
    row_heights = [0.5]  # Coverage plot takes 50% of the height
    row_heights.extend([0.5 / num_transcripts] * num_transcripts)  # Equal distribution for transcripts
    
    # Define domain for the main plots and annotation area
    label_width_percent = 0.1  # Width of the annotation area (10% of total width)
    plot_width_percent = 1 - label_width_percent  # Width of the main plot area
    
    # Create subplots with custom domain
    fig = make_subplots(
        rows=num_transcripts + 1, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=row_heights,
        subplot_titles=[""] * (num_transcripts + 1)  # No subplot titles
    )
    
    # Set up x-axis values (positions)
    x = list(range(len(ref_coverage)))
    
    # Add reference coverage with increased opacity
    fig.add_trace(
        go.Bar(
            x=x,
            y=ref_coverage,
            name="Reference",
            marker_color="rgba(0, 191, 255, 0.8)",  # deepskyblue with higher alpha
            width=1.0
        ),
        row=1, col=1
    )
    
    # Add alternative coverage with increased opacity
    fig.add_trace(
        go.Bar(
            x=x,
            y=alt_coverage,
            name="Alternative",
            marker_color="rgba(255, 69, 0, 0.8)",  # orangered with higher alpha
            width=1.0
        ),
        row=1, col=1
    )
    
    # Add variant marker
    max_height = max(ref_coverage[variant_pos], alt_coverage[variant_pos]) + 0.01 * np.max(ref_coverage)
    fig.add_trace(
        go.Scatter(
            x=[variant_pos],
            y=[max_height],
            mode="markers",
            marker=dict(
                symbol="star",
                size=12,
                color="black"
            ),
            name="Variant",
            hoverinfo="name"
        ),
        row=1, col=1
    )
    
    # Add transcript/exon plots
    for i, (tn, transcript) in enumerate(transcripts_toplot.items(), 1):
        tr_name = tn[:-1]
        tr_direction = tn[-1]
        
        # Add a light gray background for the whole region
        fig.add_trace(
            go.Scatter(
                x=[xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]],
                y=[0, 0, 1, 1, 0],
                fill="toself",
                fillcolor="rgba(220, 220, 220, 0.2)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="none"
            ),
            row=i+1, col=1
        )
        
        # Add a line for the gene body centered in the gray box (at y=0.5)
        first_exon_start = transcript[0][0]
        last_exon_end = transcript[-1][1]
        
        fig.add_trace(
            go.Scatter(
                x=[first_exon_start, last_exon_end],
                y=[0.5, 0.5],  # Center the gene line at y=0.5
                mode="lines",
                line=dict(
                    color="black",
                    width=1
                ),
                showlegend=False,
                hoverinfo="none"
            ),
            row=i+1, col=1
        )
        
        # Add direction marker at the end of the gene body
        if tr_direction == '+':
            marker_symbol = "triangle-right"
            marker_pos = last_exon_end
        else:
            marker_symbol = "triangle-left"
            marker_pos = first_exon_start
            
        fig.add_trace(
            go.Scatter(
                x=[marker_pos],
                y=[0.5],  # Center the direction marker at y=0.5
                mode="markers",
                marker=dict(
                    symbol=marker_symbol,
                    size=8,
                    color="black"
                ),
                showlegend=False,
                hoverinfo="none"
            ),
            row=i+1, col=1
        )
        
        # Add exons centered in the gray box
        for exon_start, exon_end in transcript:
            # Define exon height (centered around y=0.5)
            exon_height = 0.9  # Height of exon boxes
            y_bottom = 0.5 - exon_height/2
            y_top = 0.5 + exon_height/2
            
            fig.add_trace(
                go.Scatter(
                    x=[exon_start, exon_end, exon_end, exon_start, exon_start],
                    y=[y_bottom, y_bottom, y_top, y_top, y_bottom],
                    mode="none",  # No markers, just the fill
                    fill="toself",
                    fillcolor="rgba(70, 130, 180, 0.6)",  # steelblue with alpha
                    line=dict(width=1, color="rgba(70, 130, 180, 0.8)"),  # Thin border
                    showlegend=False,
                    hoverinfo="none"
                ),
                row=i+1, col=1
            )
        
        # Create a frozen annotation instead of using fig.add_annotation
        # We'll add a separate subplot for annotations later
    
    # Update layout for the main plot
    fig.update_layout(
        title=f"{variant_name} coverage",
        width=fig_width,
        height=fig_height,
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=50, b=30),
        hovermode="closest",
        plot_bgcolor="rgba(240, 240, 250, 0.2)",  # Light background color
    )
    
    # Adjust x-axis domains to make room for the frozen labels
    for i in range(1, num_transcripts + 2):  # +2 because we have num_transcripts + 1 rows
        fig.update_xaxes(
            domain=[label_width_percent, 1],  # Start after the label area
            row=i,
            col=1,
            range=xlim
        )
    
    # Set y-axis range for coverage plot if specified
    if ylim is not None:
        fig.update_yaxes(range=ylim, row=1, col=1)
    else:
        # Set a reasonable default based on the data
        max_coverage = max(max(ref_coverage), max(alt_coverage))
        fig.update_yaxes(range=[0, max_coverage * 1.1], row=1, col=1)
    
    # Update y-axis title for coverage plot
    fig.update_yaxes(title_text="Coverage", row=1, col=1)
    
    # Hide x-axis title and labels for all but the last row
    for i in range(1, num_transcripts + 1):
        fig.update_xaxes(
            showticklabels=False,
            row=i,
            col=1
        )
        fig.update_yaxes(
            range=[0, 1],  # Range is now 0 to 1 to match the gray box
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            row=i+1,
            col=1
        )
    
    # Show x-axis for the last row
    fig.update_xaxes(
        title_text="Position",
        showticklabels=True,
        row=num_transcripts+1,
        col=1
    )
    
    # Now add the frozen annotation area using shapes and annotations
    # This will create a separate area that won't move when zooming
    
    # Calculate vertical positions for each transcript label
    y_positions = []
    total_height = 1.0
    coverage_height = row_heights[0] * total_height
    transcript_section_height = total_height - coverage_height
    transcript_height = transcript_section_height / num_transcripts
    
    # Calculate the bottom position of each transcript row
    for i in range(num_transcripts):
        # Position is from bottom of the plot, starting after the coverage section
        bottom_pos = 1 - coverage_height - (i+1) * transcript_height
        middle_pos = bottom_pos + transcript_height / 2
        y_positions.append(middle_pos)
    
    # Add rectangle for the frozen label area
    fig.add_shape(
        type="rect",
        x0=0,
        x1=label_width_percent,
        y0=0,
        y1=1,
        line=dict(color="rgba(0,0,0,0)"),
        fillcolor="white",
        layer="below"
    )
    
    # Add a vertical line to separate the frozen area
    fig.add_shape(
        type="line",
        x0=label_width_percent,
        x1=label_width_percent,
        y0=0,
        y1=1,
        line=dict(color="lightgrey", width=1),
        layer="below"
    )
    
    
    # Updated section for positioning annotations in the frozen area
        
    # Calculate the y-positions for each row more precisely
    y_positions = []
    row_domains = []

    # Get the y-domain information for each subplot
    for i in range(1, num_transcripts + 2):  # +2 because we have num_transcripts + 1 rows
        row_domain = fig.layout[f'yaxis{i}'].domain
        row_domains.append(row_domain)

    # The first domain is for the coverage plot, the rest are for transcripts
    coverage_domain = row_domains[0]
    transcript_domains = row_domains[1:]

    # Add the transcript names in the frozen area, aligned with their subplot
    for i, (tn, _) in enumerate(transcripts_toplot.items()):
        tr_name = tn[:-1]
        
        # Calculate the middle of this transcript's subplot domain
        domain = transcript_domains[i]
        middle_y = (domain[0] + domain[1]) / 2
        
        fig.add_annotation(
            x=label_width_percent * 0.9,  # Right-aligned in the label area
            y=middle_y,
            text=tr_name,
            showarrow=False,
            font=dict(size=8),
            xanchor="right",
            yanchor="middle",
            xref="paper",
            yref="paper"
        )

    # Create output directory if it doesn't exist
    os.makedirs(savedir, exist_ok=True)
    
    # Configure the figure for HTML output with appropriate settings
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': variant_name,
            'height': fig_height,
            'width': fig_width,
            'scale': 3  # Higher resolution
        }
    }
    
    # Save as HTML for interactive viewing
    fig.write_html(f"{savedir}/{variant_name}.html", config=config)
    
    # Save as static image (PNG and PDF)
    #fig.write_image(f"{savedir}/{variant_name}.png")
    #fig.write_image(f"{savedir}/{variant_name}.pdf")
    
    return fig


def plot_coverage_tracks(ref_coverage, alt_coverage, variant_pos, variant_name, transcripts_toplot, xlim=None, ylim=None,
                         fig_width=12, fig_height=2.5, savedir='.'):
    """Plots reference and alternative coverage tracks with variant highlight and exon annotations.

    Args:
        ref_coverage: Array of coverage values for the reference allele.
        alt_coverage: Array of coverage values for the alternative allele.
        variant_pos: Position (index) of the variant in the coverage arrays.
        transcripts_toplot: Dictionary of transcripts to plot, with transcript name as key and list of exon coordinates as value.
        xlim: Tuple of (min, max) x-axis limits.
        ylim: Tuple of (min, max) y-axis limits.
        fig_width: Width of the figure (optional).
        fig_height: Height of the figure (optional).
        savedir: Directory to save the figure to.
    """
    
    # Create figure and axes
    num_transcripts = len(transcripts_toplot.keys())
    height_ratios = [1.3]
    height_ratios.extend([0.3]*num_transcripts)
    fig_height = 1.3 + 0.3*num_transcripts
    
    fig, axes = plt.subplots(num_transcripts+1, 1, figsize=(fig_width, fig_height),
                            gridspec_kw={'height_ratios': height_ratios},
                            sharex=True)  
    
    ax_coverage = axes[0]

    # Set up x-axis values (positions)
    x = np.arange(len(ref_coverage))
    ax_coverage.bar(x, ref_coverage, color='deepskyblue', width=1.0, 
                  alpha=0.5, label='Reference', linewidth=0)
    ax_coverage.bar(x, alt_coverage, color='orangered', width=1.0, 
                  alpha=0.5, label='Alternative', linewidth=0)
    ax_coverage.plot(variant_pos, max(ref_coverage[variant_pos], alt_coverage[variant_pos]), 
                   marker='*', color='red', markersize=12, label='Variant', zorder=100)   
    
    # Formatting
    ax_coverage.set_xlabel('')
    ax_coverage.set_ylabel('Coverage')
    title_str = f'{variant_name} coverage'
    ax_coverage.set_title(title_str)
    if ylim is not None:
        ax_coverage.set_ylim(ylim)
    ax_coverage.legend()
    sns.despine()
    
    # Get xlim
    if xlim is None:
        xlim = [0, len(ref_coverage)]
        
    # Exon plot
    for ti, tn in enumerate(transcripts_toplot.keys()):
        transcript = transcripts_toplot[tn]
        tr_name = tn[:-1]
        tr_direction = tn[-1]
        
        ax_exons = axes[ti+1]
        ax_exons.axvspan(xlim[0], xlim[1], facecolor='gray', alpha=0.1)  
        
        for exon_start, exon_end in transcript:
            ax_exons.axvspan(exon_start, exon_end, facecolor='steelblue', alpha=0.6)  
            
        # Plot a gray line from the beginning of the first exon to the end of the last exon in the middle of the y axis   
        ax_exons.plot([transcript[0][0], transcript[-1][1]], [0, 0], color='k', linewidth=1, 
                    marker="4" if tr_direction == '+' else "3", markersize=9)

        ax_exons.set_xticks([])
        # Put text at the start of the x axis
        ax_exons.text(xlim[0], 0, tr_name, fontsize=8, verticalalignment='top')
        ax_exons.set_ylabel('')           
        ax_exons.set_yticks([])  # Remove y-axis ticks
        
        # Remove axis lines
        ax_exons.spines['top'].set_visible(False)
        ax_exons.spines['right'].set_visible(False)
        ax_exons.spines['bottom'].set_visible(False)
        ax_exons.spines['left'].set_visible(False)
        
        if xlim is not None:
            ax_exons.set_xlim(xlim)
        if ylim is not None:
            ax_exons.set_ylim(ylim)

    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(f'{savedir}/{variant_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def parse_seqs_del(fasta_open, start, end, chrom, seq_len, deletion, var_to, pos_indel):
    """Parse sequences for deletion variant."""
    if start < 0:
        seq_dna = "N" * (-start) + fasta_open.fetch(chrom, 0, end)
    else:
        seq_dna = fasta_open.fetch(chrom, start, end)
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    # Right shift: same on the right side; more left sequence needed because of deletion
    start_alt1 = start - deletion
    end_alt1 = end
    # Left shift: same on the left side; more right sequence needed because of deletion
    start_alt2 = start
    end_alt2 = end + deletion

    start_ref1 = start + deletion
    end_ref1 = end + deletion

    if start_ref1 < 0:
        seq_dna_right = "N" * (-start_ref1) + fasta_open.fetch(chrom, 0, end_ref1)
    else:
        seq_dna_right = fasta_open.fetch(chrom, start_ref1, end_ref1)
    if len(seq_dna_right) < seq_len:
        seq_dna_right += "N" * (seq_len - len(seq_dna_right))

    if start_alt2 < 0:
        seq_dna_alt2 = (
            "N" * (-start_alt2)
            + fasta_open.fetch(chrom, 0, pos_indel)
            + var_to
            + fasta_open.fetch(chrom, pos_indel + deletion + 1, end_alt2)
        )
    else:
        seq_dna_alt2 = (
            fasta_open.fetch(chrom, start_alt2, pos_indel)
            + var_to
            + fasta_open.fetch(chrom, pos_indel + deletion + 1, end_alt2)
        )
    if len(seq_dna_alt2) < seq_len:
        seq_dna_alt2 += "N" * (seq_len - len(seq_dna_alt2))

    return (seq_dna, seq_dna_right, seq_dna_alt2)


def parse_seqs_ins(fasta_open, start, end, chrom, seq_len, deletion, var_to, pos_indel):
    """Parse sequences for insertion variant."""
    if start < 0:
        seq_dna = "N" * (-start) + fasta_open.fetch(chrom, 0, end)
    else:
        seq_dna = fasta_open.fetch(chrom, start, end)
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    # Left shift: less sequence needed on left side
    start_alt1 = start + deletion
    end_alt1 = end
    # Right shift: less sequence needed on right side
    start_alt2 = start
    end_alt2 = end - deletion

    if start_alt1 < 0:
        seq_dna_alt1 = (
            "N" * (-start_alt1)
            + fasta_open.fetch(chrom, 0, pos_indel)
            + var_to
            + fasta_open.fetch(chrom, pos_indel + 1, end_alt1)
        )
    else:
        seq_dna_alt1 = (
            fasta_open.fetch(chrom, start_alt1, pos_indel)
            + var_to
            + fasta_open.fetch(chrom, pos_indel + 1, end_alt1)
        )
    if len(seq_dna_alt1) < seq_len:
        seq_dna_alt1 += "N" * (seq_len - len(seq_dna_alt1))

    if start_alt2 < 0:
        seq_dna_alt2 = (
            "N" * (-start_alt2)
            + fasta_open.fetch(chrom, 0, pos_indel)
            + var_to
            + fasta_open.fetch(chrom, pos_indel + 1, end_alt2)
        )
    else:
        seq_dna_alt2 = (
            fasta_open.fetch(chrom, start_alt2, pos_indel)
            + var_to
            + fasta_open.fetch(chrom, pos_indel + 1, end_alt2)
        )
    if len(seq_dna_alt2) < seq_len:
        seq_dna_alt2 += "N" * (seq_len - len(seq_dna_alt2))

    return (seq_dna, seq_dna_alt1, seq_dna_alt2)


def predict_tracks(models, sequence_one_hot, n_folds=1):
    """Predict tracks using the provided models."""
    predicted_tracks = []
    for fold_ix in range(n_folds):
        yh = models[fold_ix](sequence_one_hot[None, ...])[:, None, ...].astype("float16")
        predicted_tracks.append(yh)

    predicted_tracks = np.concatenate(predicted_tracks, axis=1)
    return predicted_tracks


def stitch_preds(preds, shifts=[0]):
    """Stitch indel left and right compensation shifts.

    Args:
        preds [np.array]: List of predictions.
        shifts [int]: List of shifts.
    """
    cp = preds[0].shape[0] // 2
    preds_stitch = []
    for hi, shift in enumerate(shifts):
        hil = 2 * hi
        hir = hil + 1
        preds_stitch_i = np.concatenate((preds[hil][:cp], preds[hir][cp:]), axis=0)
        preds_stitch.append(preds_stitch_i)
    return preds_stitch


def create_temp_folder():
    """Create a temporary folder for intermediate files."""
    temp_folder = 'temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
    return temp_folder


def find_bedtools_path():
    """Find the path to bedtools executables."""
    proc = subprocess.Popen(["which", "shuffleBed"], stdout=subprocess.PIPE)
    out = proc.stdout.read().decode("utf-8")
    bedtools_exec = "/".join(out.strip("\n").split("/")[:-1])
    print("bedtools executable path to be used:", bedtools_exec)
    return bedtools_exec


def intersect_with_genes(temp_folder, bedtools_exec, pred_bed_path, gencode_bed_path):
    """Intersect prediction region with genes."""
    output_file = f"{temp_folder}/genes_intersect.bed"
    with open(output_file, "w") as f:
        subprocess.call(
            [
                f"{bedtools_exec}/intersectBed",
                "-a",
                gencode_bed_path,
                "-b",
                pred_bed_path,
                "-wa",
            ],
            stdout=f,
        )
    return output_file


def create_pred_bed(temp_folder, chrn, start, end):
    """Create BED file for prediction region."""
    pred_bed = pd.DataFrame({'chr': [chrn], 'st': [start], 'en': [end]})
    pred_bed_path = f"{temp_folder}/pred.bed"
    pred_bed.to_csv(pred_bed_path, sep="\t", header=False, index=False)
    return pred_bed_path


def get_transcript_list(df_gene):
    """Get list of transcripts from gene dataframe."""
    list_transcripts = []
    for sts, ens in zip(df_gene['exonStarts'], df_gene['exonEnds']):
        sts_split = [int(x) for x in sts.split(',')[:-1]]
        ens_split = [int(x) for x in ens.split(',')[:-1]]
        list_transcripts.append([(int(x), int(y)) for x, y in zip(sts_split, ens_split)])
    return list_transcripts


def get_transcripts_slices(df_gene, list_transcripts, seq_out_st, seq_out_en, var_from, var_to, len_indel, bins_toplot, xlim=None):
    """Get transcript slices for plotting."""
    transcripts_slices = {}
    variant_loc = (seq_out_st + seq_out_en) // 2
    
    for tr_name, strand, transcript in zip(df_gene['name_full'], df_gene['strand'], list_transcripts):
        exons_plot = []
        for exon in transcript:
            if exon[1] > seq_out_st and exon[0] < seq_out_en:
                exon_begin = exon[0]
                exon_end = exon[1]
                # deletion
                if var_from > var_to:
                    if exon_begin > variant_loc + len_indel:
                        # shift to the left
                        exon_begin = exon_begin - len_indel
                        exon_end = exon_end - len_indel
                    elif exon_begin < variant_loc and exon_end > variant_loc + len_indel:
                        # deletion in the exon
                        exon_end = exon_end - len_indel
                    elif exon_begin > variant_loc and exon_begin < variant_loc + len_indel and exon_end > variant_loc + len_indel:
                        # deletion overlaps with the exon
                        exon_begin = variant_loc
                        exon_end = exon_end - len_indel
                
                if var_from < var_to:
                    if exon_begin > variant_loc + len_indel:
                        # shift to the right
                        exon_begin = exon_begin + len_indel
                        exon_end = exon_end + len_indel
                    elif exon_begin < variant_loc and exon_end > variant_loc:
                        # insertion in the exon
                        exon_end = exon_end + len_indel

                slice_start = int((exon_begin - seq_out_st) / 32)
                slice_end = int((exon_end - seq_out_st) / 32)
                slice_start = max(0, slice_start)
                slice_end = min(slice_end, bins_toplot)
                
                if xlim is not None:
                    if slice_start > xlim[1] or slice_end < xlim[0]:
                        continue
                        
                exons_plot.append((slice_start, slice_end))
                
        if len(exons_plot) > 0:
            transcripts_slices[tr_name + strand] = exons_plot
            
    return transcripts_slices


def get_sliced_coverage(rp_snp, ap_snp, bins_full, bins_toplot, index_targets):
    """Get sliced coverage for plotting."""
    ref_track = np.mean(rp_snp[0, :, index_targets], axis=0)
    alt_track_left = np.mean(ap_snp[0, :, index_targets], axis=0)

    ref_coverage = ref_track[(bins_full-bins_toplot)//2:(bins_full+bins_toplot)//2]
    alt_coverage = alt_track_left[(bins_full-bins_toplot)//2:(bins_full+bins_toplot)//2]
    
    return ref_coverage, alt_coverage


def get_transcript_coverage(transcripts_slices_full, transcript_name, rp_snp, ap_snp, index_targets):
    """Get coverage for a specific transcript."""
    exons = transcripts_slices_full[transcript_name]
    
    # Make arrays of indices from slices for each transcript
    ind_toslice = []
    for exon in exons:
        ind_toslice.extend(list(range(exon[0], exon[1]+1)))
    
    ref_slice = rp_snp[:, ind_toslice, :]
    ref_slice = np.sum(ref_slice[:, :, index_targets], axis=1)
    
    alt_slice = ap_snp[:, ind_toslice, :]
    alt_slice = np.sum(alt_slice[:, :, index_targets], axis=1)
    
    return ref_slice, alt_slice