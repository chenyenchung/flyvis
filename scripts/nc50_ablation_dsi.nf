#!/usr/bin/env nextflow
nextflow.enable.types = true
nextflow.enable.strict = true

/*
 * NC50 ablation workflow for the flyvis flow/0000 ensemble.
 *
 * Stages:
 *   1. Create edge-mask ablated network directories for each NC50 type/model.
 *   2. Recompute task error into validation/ for each ablated network.
 *   3. Rank WT and ablated ensembles independently and compute T4/T5 DSI stats.
 *
 * Example:
 *   nextflow run scripts/nc50_ablation_dsi.nf -resume \
 *     --repo /scratch/ycc520/flyvis \
 *     --base_ensemble /scratch/ycc520/flyvis/data/results/flow/0000
 */

params {
  repo: Path = '/scratch/ycc520/flyvis'
  base_ensemble: Path = '/scratch/ycc520/flyvis/data/results/flow/0000'
  out_dir: String = '/scratch/ycc520/flyvis/data/results/flow/0000_nc50_dsi'

  sif: Path = '/scratch/ycc520/flyvis/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif'
  overlay: Path = '/scratch/ycc520/flyvis/flyvis.ext3'
  container_cmd: String = 'apptainer'
  container_flags_cpu: String = ''
  container_flags_gpu: String = '--nv'

  types: String = 'Mi1,Mi10,Mi4,Mi9,Tm1,Tm2,Tm20,Tm3,Tm5a,Tm5b,Tm5c,Tm9,TmY5a'
  n_models: Integer = 50
  top_n: Integer = 20
  stride_u: Integer = 2
  stride_v: Integer = 1

  validation_subdir: String = 'validation'
  wt_validation_subdir: String = 'validation'
  loss_file_name: String = 'epe'

  validate_skip_existing: Boolean = true
  ablate_skip_existing: Boolean = false
  skip_plots: Boolean = false
}

process PrepareAblatedNetwork {
  cpus '1'
  memory '8GB'
  time '45m'

  input:
  tuple(cell_type: String, model_id: String)

  output:
  done: Path = file("ablation_${cell_type}_${model_id}.done")

  script:
  """
  set -euo pipefail

  BASE="${params.base_ensemble}"
  TYPE="${cell_type}"
  NID="${model_id}"
  TARGET="\$(dirname "\${BASE}")/\$(basename "\${BASE}")_\${TYPE}/\${NID}"

  if [[ "${params.ablate_skip_existing}" == "true" \
        && -f "\${TARGET}/chkpts/chkpt_00000" \
        && -f "\${TARGET}/best_chkpt_index.h5" \
        && -f "\${TARGET}/chkpt_index.h5" ]]; then
    echo "Ablation already exists for \${TYPE}/\${NID}: \${TARGET}"
  else
    ${params.container_cmd} exec ${params.container_flags_cpu} \
      --overlay ${params.overlay}:ro \
      ${params.sif} \
      bash -lc "set -euo pipefail
        cd ${params.repo}
        export MAMBA_ROOT_PREFIX=/ext3/mamba
        export CONDARC=/ext3/.condarc
        export PATH=/ext3/bin:\${PATH}
        export VIRTUAL_CLUSTER=true
        micromamba run -n flyvis python scripts/ablate_network.py \
          ${params.base_ensemble}/${model_id} \
          --ablate ${cell_type} \
          --stride-u ${params.stride_u} \
          --stride-v ${params.stride_v}"
  fi

  test -f "\${TARGET}/chkpts/chkpt_00000"
  test -f "\${TARGET}/best_chkpt_index.h5"
  test -f "\${TARGET}/chkpt_index.h5"

  {
    echo "type=\${TYPE}"
    echo "model_id=\${NID}"
    echo "network_dir=\${TARGET}"
  } > ablation_${cell_type}_${model_id}.done
  """

  stub:
  """
  BASE="${params.base_ensemble}"
  TYPE="${cell_type}"
  NID="${model_id}"
  TARGET="\$(dirname "\${BASE}")/\$(basename "\${BASE}")_\${TYPE}/\${NID}"

  {
    echo "type=\${TYPE}"
    echo "model_id=\${NID}"
    echo "network_dir=\${TARGET}"
  } > ablation_${cell_type}_${model_id}.done
  """
}

process ValidateAblatedNetwork {
  cpus '1'
  memory '8GB'
  time '45m'

  input:
  ablation_done: Path

  output:
  done: Path = file("validation_*.done")

  script:
  """
  set -euo pipefail

  TYPE="\$(awk -F= '\$1 == "type" { print \$2 }' ${ablation_done})"
  NID="\$(awk -F= '\$1 == "model_id" { print \$2 }' ${ablation_done})"
  NETWORK_DIR="\$(awk -F= '\$1 == "network_dir" { print \$2 }' ${ablation_done})"

  if [[ "${params.validate_skip_existing}" == "true" \
        && -f "\${NETWORK_DIR}/${params.validation_subdir}/${params.loss_file_name}.h5" ]]; then
    echo "Validation already exists for \${TYPE}/\${NID}"
  else
    ${params.container_cmd} exec ${params.container_flags_cpu} \
      --overlay ${params.overlay}:ro \
      ${params.sif} \
      bash -lc "set -euo pipefail
        cd ${params.repo}
        export MAMBA_ROOT_PREFIX=/ext3/mamba
        export CONDARC=/ext3/.condarc
        export PATH=/ext3/bin:\${PATH}
        export VIRTUAL_CLUSTER=true
        micromamba run -n flyvis python scripts/validate_ablation_task_error.py \
          --base ${params.base_ensemble} \
          --types \${TYPE} \
          --network-ids \${NID} \
          --validation-subdir ${params.validation_subdir} \
          --skip-existing"
  fi

  test -f "\${NETWORK_DIR}/${params.validation_subdir}/${params.loss_file_name}.h5"

  {
    echo "type=\${TYPE}"
    echo "model_id=\${NID}"
    echo "network_dir=\${NETWORK_DIR}"
    echo "validation_file=\${NETWORK_DIR}/${params.validation_subdir}/${params.loss_file_name}.h5"
  } > validation_\${TYPE}_\${NID}.done
  """

  stub:
  """
  TYPE="\$(awk -F= '\$1 == "type" { print \$2 }' ${ablation_done})"
  NID="\$(awk -F= '\$1 == "model_id" { print \$2 }' ${ablation_done})"
  NETWORK_DIR="\$(awk -F= '\$1 == "network_dir" { print \$2 }' ${ablation_done})"

  {
    echo "type=\${TYPE}"
    echo "model_id=\${NID}"
    echo "network_dir=\${NETWORK_DIR}"
    echo "validation_file=\${NETWORK_DIR}/${params.validation_subdir}/${params.loss_file_name}.h5"
  } > validation_\${TYPE}_\${NID}.done
  """
}

process AssessNC50DSI {
  cpus '1'
  memory '32GB'
  time '12h'
  // clusterOptions '--gres=gpu:1'

  input:
  validation_done: Bag<Path>

  output:
  outdir: Path = file('nc50_dsi', type: 'dir')

  script:
  """
  set -euo pipefail

  ${params.container_cmd} exec ${params.container_flags_gpu} \
    --overlay ${params.overlay}:ro \
    ${params.sif} \
    bash -lc "set -euo pipefail
      cd ${params.repo}
      export MAMBA_ROOT_PREFIX=/ext3/mamba
      export CONDARC=/ext3/.condarc
      export PATH=/ext3/bin:\${PATH}
      export VIRTUAL_CLUSTER=true
      micromamba run -n flyvis python scripts/assess_nc50_dsi.py \
        --base ${params.base_ensemble} \
        --types ${params.types.replace(',', ' ')} \
        --top-n ${params.top_n} \
        --wt-validation-subdir ${params.wt_validation_subdir} \
        --ablation-validation-subdir ${params.validation_subdir} \
        --loss-file-name ${params.loss_file_name} \
        --out ${params.out_dir} \
        ${params.skip_plots ? '--skip-plots' : ''}"

  rm -rf nc50_dsi
  mkdir -p nc50_dsi
  cp -a ${params.out_dir}/. nc50_dsi/
  """

  stub:
  """
  mkdir -p nc50_dsi
  touch nc50_dsi/rankings.csv
  touch nc50_dsi/dsi_long.csv
  touch nc50_dsi/stats_summary.csv
  touch nc50_dsi/same_id_supplemental.csv
  """
}

process VisualizeAblationEPE {
  cpus '1'
  memory '8GB'
  time '1h'

  input:
  validation_done: Bag<Path>
  dsi_outdir: Path

  output:
  outdir: Path = file('nc50_dsi', type: 'dir')

  script:
  """
  set -euo pipefail

  mkdir -p ${params.out_dir}
  cp -a ${dsi_outdir}/. ${params.out_dir}/

  cat > visualize_ablation_epe.py <<'PY'
from pathlib import Path
import sys

repo = Path('${params.repo}')
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / 'scripts'))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

from assess_nc50_dsi import build_rankings
from nc50_utils import parse_type_selection

base = Path('${params.base_ensemble}')
out_dir = Path('${params.out_dir}')
ablation_types = parse_type_selection('${params.types.replace(',', ' ')}'.split())
groups = ['WT', *ablation_types]

rankings = build_rankings(
    base,
    ablation_types=ablation_types,
    top_n=${params.top_n},
    wt_validation_subdir='${params.wt_validation_subdir}',
    ablation_validation_subdir='${params.validation_subdir}',
    loss_file_name='${params.loss_file_name}',
)

epe_table = rankings.rename(columns={'loss': 'epe'}).copy()
out_dir.mkdir(parents=True, exist_ok=True)
epe_table.to_csv(out_dir / 'epe_by_ablation.csv', index=False)

def finite_stat(values, fn):
    values = np.asarray(values.dropna(), dtype=float)
    if values.size == 0:
        return np.nan
    return float(fn(values))

summary_rows = []
for group in groups:
    group_table = epe_table[epe_table['group'] == group]
    selected_table = group_table[group_table['selected']]
    values = group_table['epe']
    selected_values = selected_table['epe']
    summary_rows.append(
        {
            'group': group,
            'ablation_type': 'WT' if group == 'WT' else group,
            'n_models': int(len(group_table)),
            'n_selected': int(len(selected_table)),
            'mean_epe': finite_stat(values, np.mean),
            'median_epe': finite_stat(values, np.median),
            'min_epe': finite_stat(values, np.min),
            'max_epe': finite_stat(values, np.max),
            'selected_mean_epe': finite_stat(selected_values, np.mean),
            'selected_median_epe': finite_stat(selected_values, np.median),
            'selected_min_epe': finite_stat(selected_values, np.min),
            'selected_max_epe': finite_stat(selected_values, np.max),
        }
    )

pd.DataFrame(summary_rows).to_csv(
    out_dir / 'epe_summary_by_ablation.csv',
    index=False,
)

plot_groups = []
plot_data = []
for group in groups:
    values = epe_table.loc[epe_table['group'] == group, 'epe'].dropna().to_numpy()
    if values.size:
        plot_groups.append(group)
        plot_data.append(values)

if plot_data:
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(plot_groups)), 3.2))
    ax.violinplot(plot_data, showmeans=True, showmedians=True)
    for xpos, values in enumerate(plot_data, start=1):
        ordered = np.sort(values)
        if ordered.size == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.12, 0.12, ordered.size)
        ax.scatter(
            xpos + jitter,
            ordered,
            s=12,
            alpha=0.55,
            linewidths=0,
            color='black',
        )
    ax.set_xticks(range(1, len(plot_groups) + 1))
    ax.set_xticklabels(plot_groups, rotation=60, ha='right')
    ax.set_ylabel('Minimum validation EPE')
    ax.set_title('Validation EPE by Ablation')
    fig.tight_layout()
    fig.savefig(out_dir / 'epe_violin_by_ablation.png', dpi=200)
    plt.close(fig)
PY

  ${params.container_cmd} exec ${params.container_flags_cpu} \
    --overlay ${params.overlay}:ro \
    ${params.sif} \
    bash -lc "set -euo pipefail
      export MAMBA_ROOT_PREFIX=/ext3/mamba
      export CONDARC=/ext3/.condarc
      export PATH=/ext3/bin:\${PATH}
      export PYTHONPATH=${params.repo}:${params.repo}/scripts:\${PYTHONPATH:-}
      export VIRTUAL_CLUSTER=true
      micromamba run -n flyvis python visualize_ablation_epe.py"

  rm -rf nc50_dsi
  mkdir -p nc50_dsi
  cp -a ${params.out_dir}/. nc50_dsi/
  """

  stub:
  """
  rm -rf nc50_dsi
  mkdir -p nc50_dsi
  cp -a ${dsi_outdir}/. nc50_dsi/ 2>/dev/null || true
  touch nc50_dsi/epe_by_ablation.csv
  touch nc50_dsi/epe_summary_by_ablation.csv
  touch nc50_dsi/epe_violin_by_ablation.png
  """
}

workflow {

  main:
  type_ch = channel.fromList(
    params.types
      .split(',')
      .collect { it.trim() }
      .findAll { it }
  )

  model_ch = channel.fromList(
    (0..<params.n_models).collect { String.format('%03d', it as int) }
  )

  ablation_jobs_ch = type_ch
    .combine(model_ch)
    .map { cell_type, model_id -> tuple(cell_type, model_id) }

  ablation_done_ch = PrepareAblatedNetwork(ablation_jobs_ch)
  validation_done_ch = ValidateAblatedNetwork(ablation_done_ch)
  validation_done_bag_ch = validation_done_ch.collect()
  dsi_outdir_ch = AssessNC50DSI(validation_done_bag_ch)
  VisualizeAblationEPE(validation_done_bag_ch, dsi_outdir_ch)
}
