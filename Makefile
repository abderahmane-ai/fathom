PYTHON ?= uv run python
ARTIFACT_ROOT ?= /artifacts
RESULTS_DIR ?= results
PLOTS_DIR ?= plots
AGGREGATE_DIR ?= $(RESULTS_DIR)/aggregate

BENCHMARKS ?= ablation,depth_needle,depth_preservation,inference_memory,iso_flop,lm_quality,natural_niah,scaling_efficiency

.PHONY: help ingest plots tables $(PLOTS_DIR) $(RESULTS_DIR) summary report clean

help:
	@echo "Make targets:"
	@echo "  make ingest   - Walk the artifact root and emit per-benchmark CSVs (scripts/ingest/collect.py)"
	@echo "  make plots    - Generate publication-quality PNG+PDF plots (scripts/plots/*)"
	@echo "  make tables   - Generate per-benchmark markdown summaries (scripts/tables/*)"
	@echo "  make summary  - Render the top-level SUMMARY.md from per-benchmark ones (scripts/render_summary.py)"
	@echo "  make report   - Full pipeline: ingest + plots + tables + summary"
	@echo "  make clean    - Remove the generated results/ and plots/ directories"
	@echo ""
	@echo "Variables (override with VAR=value):"
	@echo "  ARTIFACT_ROOT (default: $(ARTIFACT_ROOT))  - Where the Modal volume is mounted"
	@echo "  RESULTS_DIR   (default: $(RESULTS_DIR))      - Where to put SUMMARY.md and per-benchmark tables"
	@echo "  PLOTS_DIR     (default: $(PLOTS_DIR))        - Where to put the PNG/PDF plots"
	@echo "  BENCHMARKS    (default: $(BENCHMARKS))       - Comma-separated list of benchmarks"

ingest:
	@mkdir -p $(AGGREGATE_DIR)
	$(PYTHON) -m scripts.ingest.collect --root $(ARTIFACT_ROOT) --out $(AGGREGATE_DIR) --benchmarks $(BENCHMARKS)

plots: $(PLOTS_DIR)

$(PLOTS_DIR):
	@mkdir -p $(PLOTS_DIR)
	@if [ -f $(AGGREGATE_DIR)/depth_preservation.csv ]; then \
	  $(PYTHON) -m scripts.plots.dps_curves --csv $(AGGREGATE_DIR)/depth_preservation.csv \
	    --artifact-root $(ARTIFACT_ROOT) --out $(PLOTS_DIR)/depth_preservation || true; \
	fi
	@if [ -f $(AGGREGATE_DIR)/scaling_efficiency_summary.csv ]; then \
	  $(PYTHON) -m scripts.plots.scaling_pareto \
	    --csv $(AGGREGATE_DIR)/scaling_efficiency_summary.csv \
	    --steps-csv $(AGGREGATE_DIR)/scaling_efficiency_steps.csv \
	    --out $(PLOTS_DIR)/scaling_efficiency || true; \
	fi
	@if [ -f $(AGGREGATE_DIR)/inference_memory.csv ]; then \
	  $(PYTHON) -m scripts.plots.inference_memory \
	    --csv $(AGGREGATE_DIR)/inference_memory.csv \
	    --out $(PLOTS_DIR)/inference_memory || true; \
	fi
	@if [ -f $(AGGREGATE_DIR)/iso_flop_summary.csv ]; then \
	  $(PYTHON) -m scripts.plots.iso_flop \
	    --csv $(AGGREGATE_DIR)/iso_flop_summary.csv \
	    --steps-csv $(AGGREGATE_DIR)/iso_flop_steps.csv \
	    --out $(PLOTS_DIR)/iso_flop || true; \
	fi
	@if [ -f $(AGGREGATE_DIR)/lm_quality_steps.csv ]; then \
	  $(PYTHON) -m scripts.plots.loss_curves \
	    --csv $(AGGREGATE_DIR)/lm_quality_steps.csv \
	    --out $(PLOTS_DIR)/lm_quality || true; \
	fi

tables: $(RESULTS_DIR)

$(RESULTS_DIR):
	@mkdir -p $(RESULTS_DIR)
	@# depth_preservation: dps.csv -> dps_table
	@if [ -f $(AGGREGATE_DIR)/depth_preservation.csv ]; then \
	  $(PYTHON) -m scripts.tables.dps_table \
	    --csv $(AGGREGATE_DIR)/depth_preservation.csv \
	    --out $(RESULTS_DIR)/depth_preservation/SUMMARY.md || true; \
	fi
	@# inference_memory: no per-mode table (already structured in the CSV)
	@# natural_niah: niah.csv -> niah_table
	@if [ -f $(AGGREGATE_DIR)/natural_niah.csv ]; then \
	  $(PYTHON) -m scripts.tables.niah_table \
	    --csv $(AGGREGATE_DIR)/natural_niah.csv \
	    --out $(RESULTS_DIR)/natural_niah/SUMMARY.md || true; \
	fi
	@# iso_flop: iso_flop_summary.csv + iso_flop_steps.csv -> iso_flop_table
	@if [ -f $(AGGREGATE_DIR)/iso_flop_summary.csv ]; then \
	  $(PYTHON) -m scripts.tables.iso_flop_table \
	    --csv $(AGGREGATE_DIR)/iso_flop_summary.csv \
	    --steps-csv $(AGGREGATE_DIR)/iso_flop_steps.csv \
	    --out $(RESULTS_DIR)/iso_flop/SUMMARY.md || true; \
	fi
	@# ablation: ablation_summary.csv + ablation_steps.csv -> ablation_table
	@if [ -f $(AGGREGATE_DIR)/ablation_summary.csv ]; then \
	  $(PYTHON) -m scripts.tables.ablation_table \
	    --csv $(AGGREGATE_DIR)/ablation_summary.csv \
	    --steps-csv $(AGGREGATE_DIR)/ablation_steps.csv \
	    --out $(RESULTS_DIR)/ablation/SUMMARY.md || true; \
	fi
	@# depth_needle: depth_needle_summary.csv + depth_needle_steps.csv -> use lm_quality_table
	@# (same shape: per-mode val_loss / throughput).  We could add a dedicated
	# depth_needle_table later if the metric set diverges.
	@if [ -f $(AGGREGATE_DIR)/depth_needle_summary.csv ]; then \
	  $(PYTHON) -m scripts.tables.lm_quality_table \
	    --csv $(AGGREGATE_DIR)/depth_needle_summary.csv \
	    --steps-csv $(AGGREGATE_DIR)/depth_needle_steps.csv \
	    --out $(RESULTS_DIR)/depth_needle/SUMMARY.md || true; \
	fi
	@# lm_quality: lm_quality_summary.csv + lm_quality_steps.csv -> lm_quality_table
	@if [ -f $(AGGREGATE_DIR)/lm_quality_summary.csv ]; then \
	  $(PYTHON) -m scripts.tables.lm_quality_table \
	    --csv $(AGGREGATE_DIR)/lm_quality_summary.csv \
	    --steps-csv $(AGGREGATE_DIR)/lm_quality_steps.csv \
	    --out $(RESULTS_DIR)/lm_quality/SUMMARY.md || true; \
	fi
	@# scaling_efficiency: scaling_efficiency_summary.csv + scaling_efficiency_steps.csv
	@if [ -f $(AGGREGATE_DIR)/scaling_efficiency_summary.csv ]; then \
	  $(PYTHON) -m scripts.tables.lm_quality_table \
	    --csv $(AGGREGATE_DIR)/scaling_efficiency_summary.csv \
	    --steps-csv $(AGGREGATE_DIR)/scaling_efficiency_steps.csv \
	    --out $(RESULTS_DIR)/scaling_efficiency/SUMMARY.md || true; \
	fi

summary:
	$(PYTHON) -m scripts.render_summary --results $(RESULTS_DIR) --out $(RESULTS_DIR)/SUMMARY.md

report: ingest plots tables summary
	@echo ""
	@echo "Report generated. See:"
	@echo "  - $(RESULTS_DIR)/SUMMARY.md           (top-level summary)"
	@echo "  - $(RESULTS_DIR)/<benchmark>/SUMMARY.md (per-benchmark)"
	@echo "  - $(PLOTS_DIR)/<benchmark>/             (PNG + PDF plots)"

clean:
	rm -rf $(RESULTS_DIR) $(PLOTS_DIR)
