# %%
import time

_TIMINGS = []  # (section label, elapsed seconds)

def tic():
    """Start a timer; returns a wall-clock timestamp in seconds."""
    return time.time()

def toc(t0, label):
    """Stop a timer started with tic(), report it, and record it for the summary."""
    elapsed = time.time() - t0
    _TIMINGS.append((label, elapsed))
    print(f"[timing] {label}: {elapsed:.3f} s")
    return elapsed


t0 = tic()  # start timing: imports

from typing_extensions import ParamSpecArgs
import numpy as np
import pandas as pd
import scanpy as sc

import sys
import argparse
import cProfile

toc(t0, "imports")

# %%
t0 = tic()  # start timing: scanpy settings

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
# sc.settings.n_jobs = int(sys.argv[4])
sc.settings.n_jobs = 1

print(f"using {sc.settings.n_jobs} threads")

toc(t0, "scanpy settings")

# %%
t0 = tic()  # start timing: argument parsing

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--data-dir', type=str, help='Directory containing the dataset subdirectories', default='data')
parser.add_argument('--data-set', type=str, help='Dataset name, which is the subdirectory name', default='pbmc3k')
parser.add_argument('--out-dir', type=str, help='Output directory', required=False, default='data')
parser.add_argument('--num-threads', type=int, help='Number of threads', default=1, required=False)

args = parser.parse_args()

datadir = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
dataset = args.data_set 
outdir = args.out_dir if args.out_dir.endswith('/') else args.out_dir + '/'
nthreads = args.num_threads

toc(t0, "argument parsing")

#%%

t0 = tic()  # start timing: I/O (read 10x mtx)

# I/O
results_file = "/".join([outdir, dataset + '.scanpy.h5ad'])  # the file that will store the analysis results

adata = sc.read_10x_mtx(
    #'/nethome/tpan7/scgc/data/' + dataset + '/filtered_gene_bc_matrices/hg19',  # the directory with the `.mtx` file
    "/".join([datadir, dataset, 'filtered_gene_bc_matrices']),  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)                              # write a cache file for faster subsequent reading

adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`

toc(t0, "I/O")


# %%

t0 = tic()  # start timing: filtering

# preprocessing

# basic filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

toc(t0, "filtering")

#%%
t0 = tic()  # start timing: normalization

# metric
#adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
#sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# filtering by slicing the AnnData object
#adata = adata[adata.obs.n_genes_by_counts < 2500, :]
#adata = adata[adata.obs.pct_counts_mt < 5, :]


# and normalize to 10K reads per cell
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

toc(t0, "normalization")

# %%

t0 = tic()  # start timing: highly variable genes


# highly variable genes

#sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

# freeze data.
adata.raw = adata

# filtering by highly variable genes.
adata = adata[:, adata.var.highly_variable]

toc(t0, "highly variable genes")

#%%
t0 = tic()  # start timing: scale

# regres out effects of total counts per cell an d% mitochondrial genes
#sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata)

toc(t0, "scale")

# %%
# report adata - so we can check ot see if we are comparable to Seurat
# adata.write(results_file)
# adata

# %%
t0 = tic()  # start timing: PCA

# pca.  parallel via OMP_NUM_THREADS
sc.tl.pca(adata, svd_solver='arpack', n_comps=30)

# adata.write(results_file)
# adata

toc(t0, "PCA")

# %%

t0 = tic()  # start timing: neighborhood graph

# neighborhood graph
sc.pp.neighbors(adata, n_pcs=30)

toc(t0, "neighbors")

# %% 

t0 = tic()  # start timing: clustering

# for fixing disconnected clusters or connectivity issues:
#sc.tl.paga(adata)
#sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
#cs.tl.umap(adata, init_pos='paga')


# adata.write(results_file)
# adata


# %%
# clustering  (currently uses leiden,  previously using louvain (like Seurat).)
#sc.tl.leiden(adata)
sc.tl.louvain(adata, resolution = 0.5)

toc(t0, "louvain clustering")

#%%

t0 = tic()  # start timing: UMAP

# umap
sc.tl.umap(adata, n_components=30)

toc(t0, "UMAP")

#%%

t0 = tic()  # start timing: write results

adata.write(results_file)
adata

toc(t0, "write h5ad")

# %%
t0 = tic()  # start timing: marker genes

# support t-test, wilcoxon, logistic regression
# find marker genes

profiler = cProfile.Profile()
profiler.enable()
sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', use_raw=True)
profiler.disable()

toc(t0, "rank_genes_groups")

profile_file = f"{dataset}_rank_genes_groups.prof"
profiler.dump_stats(profile_file)
print(f"[cProfile] saved to {profile_file}")


# %%
# profiling summary: elapsed time per instrumented section.
print("\n=== elapsed time per section ===")
_total = 0.0
for _label, _elapsed in _TIMINGS:
    _total += _elapsed
    print(f"{_elapsed:8.3f} s  {_label}")
print(f"{_total:8.3f} s  TOTAL (instrumented sections)")

