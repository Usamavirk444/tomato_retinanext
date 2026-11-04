# ─────────────────────────────────────────────────────────────────────
#  RetinaNeXt — shared setup: palette, theme, loaders, export
#
#  Reads the original result CSVs in R_diagrams/source_csv/. Nothing is
#  hand-entered: every number a figure shows is read from the file the
#  evaluation wrote.
#
#  Sourced by every figure script. Open any of them and press Source.
# ─────────────────────────────────────────────────────────────────────

pkgs <- c("ggplot2", "dplyr", "tidyr", "readr", "scales", "patchwork", "svglite")
need <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(need)) {
  message("installing: ", paste(need, collapse = ", "))
  install.packages(need, repos = "https://cloud.r-project.org")
}
invisible(lapply(pkgs, library, character.only = TRUE))

set.seed(42)   # any sampling below is reproducible

# ── paths ────────────────────────────────────────────────────────────
# Resolve the project relative to this file where possible, so the folder
# can be moved or shared.
# Portable resolution of this script's own directory.
# sys.frame(1)$ofile is set when the file is source()d (RStudio "Source",
# run_all.R) but NOT under `Rscript file.R`. The previous fallback was an
# absolute path into tomato_vm_clean/, the read-only research archive, which
# does not exist on other machines and which this script must never write to.
.script_dir <- function() {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
}
.this_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA)
BASE <- if (!is.na(.this_file)) dirname(.this_file) else .script_dir()
PROJ <- dirname(BASE)

SRC  <- file.path(BASE, "source_csv")
RES  <- file.path(SRC, "results")
XD   <- file.path(SRC, "cross_eval_3_clean")   # overlap-free external set (805 images)
XDA  <- file.path(SRC, "cross_eval_3_clean")   # all stages are in the clean folder
CT   <- file.path(SRC, "results_cleantest")
LOGS <- file.path(SRC, "logs")
DATA <- file.path(PROJ, "data", "clean_yolo_dataset")   # labels, 98 MB, not copied

FIGS <- file.path(BASE, "figures")
dir.create(FIGS, showWarnings = FALSE, recursive = TRUE)

if (!dir.exists(SRC))
  stop("source_csv/ not found at ", SRC,
       "\nThe figure scripts cannot run without it.")

SEEDS <- c(42, 123, 456)

# ── canonical names ──────────────────────────────────────────────────
# One spelling per class, used everywhere. The confusion-matrix CSVs
# abbreviate Yellow Leaf Curl Virus to YLCV; every other file spells it
# out. canon_class() reconciles them so no figure shows two spellings.
CLASS_LEVELS <- c("Early Blight", "Healthy", "Late Blight", "Leaf Miners",
                  "Leaf Mold", "Mosaic Virus", "Septoria", "Spider Mites",
                  "YLCV")   # Yellow Leaf Curl Virus, abbreviated in every figure

canon_class <- function(x) {
  x <- as.character(x)
  x[x == "Yellow Leaf Curl Virus"] <- "YLCV"
  x[x == "Yellow_Leaf_Curl_Virus"] <- "YLCV"
  x[x == "Leaf Miner"]  <- "Leaf Miners"
  x
}

# Short labels for crowded axes. Full names stay in captions.
CLASS_SHORT <- c("Early Blight" = "Early Blight", "Healthy" = "Healthy",
                 "Late Blight" = "Late Blight", "Leaf Miners" = "Leaf Miners",
                 "Leaf Mold" = "Leaf Mold", "Mosaic Virus" = "Mosaic Virus",
                 "Septoria" = "Septoria", "Spider Mites" = "Spider Mites",
                 "YLCV" = "YLCV")

# ── directories, and which to trust ──────────────────────────────────
#   results/yolov8n/              Ultralytics validator. NOT used in-domain.
#   results/yolov8n_torchmetrics/ TorchMetrics, as the other three models.
# The cross-dataset run used one pipeline for every model, so the plain
# yolov8n directory is the correct source there.
#
# NOTE: source_csv/results/yolov8n/ deliberately contains ONLY
# confusion_matrix_test.csv. Its per_class_test.csv was left out because
# those mAP values come from the Ultralytics validator and disagree with
# the paper. The confusion matrices are unaffected by that and are the
# only source for YOLOv8n in Figure 9 — do not delete the folder.
DIR_IN <- c("SSDLite"                   = "ssdlite",
            "YOLOv8n"                   = "yolov8n_torchmetrics",
            "RetinaNet-R50"             = "retinanet",
            "RetinaNeXt"                = "retinanet_hybrid")
DIR_X  <- c("SSDLite"                   = "ssdlite",
            "YOLOv8n"                   = "yolov8n",
            "RetinaNet-R50"             = "retinanet",
            "RetinaNeXt"                = "retinanet_hybrid")
STAGE_DIR <- c("S1 RetinaNet-R50 (baseline)"    = "retinanet",
               "S2 ConvNeXt-T + FPN"        = "retinanet_convnext_baseline",
               "S3 ConvNeXt-T + FPN + CBAM" = "retinanet_convnext_fpn_cbam",
               "S4 ConvNeXt-T + BiFPN×3"   = "retinanet_convnext_bifpn",
               "S5 RetinaNeXt (full)"     = "retinanet_hybrid")

MODEL_LEVELS <- names(DIR_IN)
STAGE_LEVELS <- names(STAGE_DIR)

# ── palette ──────────────────────────────────────────────────────────
# Extracted from the reference figures by sampling their pixels, not
# chosen from a generic scheme. The references use two related systems:
#
#   seaborn "deep"       refs 8 and 10  (TIDE bars, cross-dataset bars)
#                        sampled: #5F759B blue, #C38C6A orange,
#                                 #AA6263 red, #5C88BC / #E1945C
#   matplotlib "Blues"   ref 7          (confusion matrices)
#                        sampled: #C9D9ED low -> #142F67 high
#
# Refs 1 and 5 use matplotlib "tab10", which is the saturated form of the
# same hues. The muted seaborn values are used here: they are what most of
# the references use, and they hold up better in print.
PAL <- c(
  blue       = "#4C72B0",   # seaborn deep blue   (ref 8/10 sampled #5C88BC)
  orange     = "#DD8452",   # seaborn deep orange (ref 10 sampled #E1945C)
  green      = "#55A868",   # seaborn deep green
  red        = "#C44E52",   # seaborn deep red    (ref 8 sampled #AA6263)
  purple     = "#8172B3",
  teal       = "#64B5CD",
  grey_dark  = "#3C3C3C",
  grey_mid   = "#8C8C8C",   # ref 10 uses grey for the comparison baseline
  grey_light = "#D8D8D8",
  # matplotlib "Blues" ramp, sampled from ref 7
  blues_low  = "#C9D9ED",
  blues_mid  = "#99BFDB",
  blues_high = "#142F67"
)

# One fixed colour per model, used in every figure.
#
# Ref 10's convention is adopted: the weakest baseline is grey, and the
# proposed model takes the orange accent. Orange against blue and green is
# the safest combination for colour-blind readers — the red/green pairing
# in ref 1 is not reproduced here, because deuteranopes cannot separate
# those two when they sit adjacent in a grouped bar chart.
MODEL_COLS <- c("SSDLite"                   = unname(PAL["grey_mid"]),
                "YOLOv8n"                   = unname(PAL["green"]),
                "RetinaNet-R50"             = unname(PAL["blue"]),
                "RetinaNeXt"                = unname(PAL["orange"]))

# Ablation: the baseline and partial configurations stay cool and muted so
# the complete model reads first without its bar being exaggerated.
STAGE_COLS <- c("S1 RetinaNet-R50 (baseline)"    = unname(PAL["grey_mid"]),
                "S2 ConvNeXt-T + FPN"        = unname(PAL["teal"]),
                "S3 ConvNeXt-T + FPN + CBAM" = unname(PAL["blue"]),
                "S4 ConvNeXt-T + BiFPN×3"   = unname(PAL["purple"]),
                "S5 RetinaNeXt (full)"     = unname(PAL["orange"]))

# Evaluation domain, fixed throughout. Matches ref 10 exactly.
DOMAIN_COLS <- c("In-domain test set" = unname(PAL["blue"]),
                 "External benchmark" = unname(PAL["orange"]))

# Error types.
ERROR_COLS <- c("Missed (no detection)"       = unname(PAL["red"]),
                "Confused with another class" = unname(PAL["purple"]))

# Sequential ramp for heatmaps and density, from ref 7's confusion matrices.
scale_fill_blues <- function(...)
  scale_fill_gradientn(colours = c("#FFFFFF", PAL[["blues_low"]],
                                   PAL[["blues_mid"]], PAL[["blues_high"]]),
                       values = c(0, 0.18, 0.55, 1), ...)

SEED_SHAPES <- c("42" = 16, "123" = 15, "456" = 17)

# ── theme ────────────────────────────────────────────────────────────
# Sized for a two-column journal page: a figure placed at ~88 mm width
# must stay legible, so base text is 10 pt and nothing smaller than 7 pt.
theme_paper <- function(base = 10) {
  theme_minimal(base_size = base, base_family = "sans") +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major   = element_line(colour = PAL[["grey_light"]],
                                        linewidth = 0.3),
      panel.background   = element_rect(fill = "white", colour = NA),
      plot.background    = element_rect(fill = "white", colour = NA),
      axis.line          = element_line(colour = PAL[["grey_dark"]],
                                        linewidth = 0.4),
      axis.ticks         = element_line(colour = PAL[["grey_dark"]],
                                        linewidth = 0.3),
      axis.text          = element_text(colour = PAL[["grey_dark"]], size = base - 1),
      axis.title         = element_text(colour = "black", size = base),
      plot.title         = element_text(face = "bold", size = base + 2,
                                        margin = margin(b = 3)),
      plot.subtitle      = element_text(colour = PAL[["grey_dark"]],
                                        size = base - 1, margin = margin(b = 8)),
      plot.caption       = element_text(colour = PAL[["grey_mid"]],
                                        size = base - 2, hjust = 0,
                                        margin = margin(t = 8)),
      legend.position    = "bottom",
      legend.title       = element_blank(),
      legend.text        = element_text(size = base - 1),
      legend.key.size    = unit(0.8, "lines"),
      strip.text         = element_text(face = "bold", size = base),
      plot.margin        = margin(6, 8, 6, 6)
    )
}

# Bars grow from zero, so no axis truncation anywhere.
scale_y_metric <- function(limit = 103, step = 20)
  scale_y_continuous(limits = c(0, limit), breaks = seq(0, 100, step),
                     expand = expansion(mult = c(0, 0.02)))

# ── in-figure text ───────────────────────────────────────────────────
# Journals typeset the caption themselves, from the manuscript. Text burned
# into the image duplicates it, cannot be edited by the typesetter, and
# will not match the journal's caption font.
#
#   TRUE   keep title / subtitle / caption   — useful while reviewing
#   FALSE  strip them                        — use this for submission
#
# Panel tags (a), (b) are NOT removed: those are referenced from the
# caption text and have to stay in the image.
IN_FIGURE_TEXT <- FALSE

# Raster resolution for the PNG copies. SVG and PDF are vector and are
# unaffected. 600 dpi is what most journals ask for on line art; drop to
# 300 if the PNGs get unwieldy. Individual calls can override with
# save_fig(..., dpi = n).
DPI <- 600

strip_text <- function(p) {
  if (IN_FIGURE_TEXT) return(p)
  if (inherits(p, "patchwork")) {
    # Clear only the plot_annotation layer. Using `&` here would also wipe
    # the per-panel subtitles, which label what each panel shows and are
    # referenced from the caption — those must stay.
    p + plot_annotation(title = NULL, subtitle = NULL, caption = NULL)
  } else {
    p + labs(title = NULL, subtitle = NULL, caption = NULL)
  }
}

# ── export ───────────────────────────────────────────────────────────
# SVG is the primary output: true vector, editable in Illustrator or
# Inkscape, and accepted by every journal that takes vector art. It is
# written with svglite rather than grDevices::svg(), because the latter
# goes through cairo, which is not present in every R install.
#
# A 300 dpi PNG is written alongside for quick viewing and for journals
# that insist on raster. PDF is produced only when cairo is available,
# since the non-cairo pdf device mangles the \u00b1 and \u00d7 characters
# used in several subtitles.
HAS_CAIRO <- isTRUE(capabilities("cairo"))
HAS_SVGLITE <- requireNamespace("svglite", quietly = TRUE)
if (!HAS_SVGLITE)
  warning("svglite not available — SVG export disabled", call. = FALSE)
if (!HAS_CAIRO)
  message("note: cairo unavailable — PDF export skipped, SVG and PNG still written")

#' @param w,h inches. Single column ~3.5, double column ~7.2.
save_fig <- function(plot, name, w = 7.2, h = 4.5, dpi = DPI) {
  plot <- strip_text(plot)
  written <- character(0)

  if (HAS_SVGLITE) {
    f <- file.path(FIGS, paste0(name, ".svg"))
    ggsave(f, plot, width = w, height = h, device = svglite::svglite, bg = "white")
    written <- c(written, "svg")
  }

  f <- file.path(FIGS, paste0(name, ".png"))
  ggsave(f, plot, width = w, height = h, dpi = dpi, bg = "white")
  written <- c(written, "png")

  if (HAS_CAIRO) {
    f <- file.path(FIGS, paste0(name, ".pdf"))
    ggsave(f, plot, width = w, height = h, device = cairo_pdf, bg = "white")
    written <- c(written, "pdf")
  }

  message("  saved ", name, "  [", paste(written, collapse = ", "),
          "]  ", w, " x ", h, " in  @ ", dpi, " dpi")
}

# ── loaders ──────────────────────────────────────────────────────────
# Each fails loudly with the missing path rather than returning a short
# frame that would silently produce an incomplete figure.

.need <- function(path, what) {
  if (!file.exists(path))
    stop(what, " not found:\n  ", path, call. = FALSE)
  path
}

read_main_results <- function() {
  f <- .need(file.path(RES, "main_results_table.csv"), "main_results_table.csv")
  lbl <- c("SSDLite-MobileNetV3-Large" = "SSDLite",
           "YOLOv8n" = "YOLOv8n",
           "RetinaNet-ResNet50-FPN-v2" = "RetinaNet-R50",
           "RetinaNet-Hybrid v3" = "RetinaNeXt")
  d <- read_csv(f, show_col_types = FALSE) %>%
    mutate(model = unname(lbl[model])) %>%
    mutate(across(where(is.numeric), ~ .x * 100))
  if (any(is.na(d$model)))
    stop("unrecognised model name in main_results_table.csv")
  d %>% mutate(model = factor(model, levels = MODEL_LEVELS))
}

read_per_class <- function() {
  out <- list()
  for (m in names(DIR_IN)) for (s in SEEDS) {
    f <- .need(file.path(RES, DIR_IN[[m]], paste0("seed_", s), "per_class_test.csv"),
               paste("per-class results for", m, "seed", s))
    d <- read_csv(f, show_col_types = FALSE)
    d <- d[!grepl("^OVERALL", d$class), ]
    out[[length(out) + 1]] <- tibble(
      model = m, seed = s, class = canon_class(d$class),
      ap50 = d$ap50 * 100,
      ap5095 = ifelse(d$ap5095 >= 0, d$ap5095 * 100, NA_real_),
      precision = ifelse(d$precision >= 0, d$precision, NA_real_),
      recall = ifelse(d$recall >= 0, d$recall, NA_real_))
  }
  bind_rows(out) %>%
    mutate(model = factor(model, levels = MODEL_LEVELS),
           class = factor(class, levels = CLASS_LEVELS))
}

.cross_file <- function(dir, s) {
  f <- file.path(XD, dir, paste0("seed_", s), "per_class_cross.csv")
  if (!file.exists(f))
    f <- file.path(XDA, dir, paste0("seed_", s), "per_class_cross.csv")
  f
}

read_per_class_cross <- function(dirs = DIR_X) {
  out <- list()
  for (m in names(dirs)) for (s in SEEDS) {
    f <- .need(.cross_file(dirs[[m]], s),
               paste("cross-dataset results for", m, "seed", s))
    d <- read_csv(f, show_col_types = FALSE)
    d <- d[!grepl("^OVERALL", d$class), ]
    out[[length(out) + 1]] <- tibble(model = m, seed = s,
                                     class = canon_class(d$class),
                                     ap50 = d$ap50 * 100,
                                     ap5095 = d$ap5095 * 100)
  }
  bind_rows(out) %>% mutate(class = factor(class, levels = CLASS_LEVELS))
}

read_cross_overall <- function(dirs = DIR_X) {
  out <- list()
  for (m in names(dirs)) for (s in SEEDS) {
    d <- read_csv(.need(.cross_file(dirs[[m]], s),
                        paste("cross-dataset results for", m)),
                  show_col_types = FALSE)
    v <- d$ap50[grepl("^OVERALL", d$class)]
    if (!length(v)) stop("no OVERALL row for ", m, " seed ", s)
    out[[length(out) + 1]] <- tibble(model = m, seed = s, map50 = v[1] * 100)
  }
  bind_rows(out)
}

read_ablation <- function() {
  mr <- read_main_results()
  out <- list()
  for (st in STAGE_LEVELS) {
    d <- STAGE_DIR[[st]]
    if (d %in% c("retinanet", "retinanet_hybrid")) {
      nm <- if (d == "retinanet") "RetinaNet-R50" else "RetinaNeXt"
      row <- mr[mr$model == nm, ]
      im <- row$map50_mean; isd <- row$map50_std
    } else {
      v <- vapply(SEEDS, function(s) {
        f <- .need(file.path(RES, d, paste0("seed_", s),
                             paste0(d, "_seed_", s, "_eval_results.csv")),
                   paste("ablation in-domain result for", st))
        read_csv(f, show_col_types = FALSE)$mAP_50[1] * 100
      }, numeric(1))
      im <- mean(v); isd <- sd(v)
    }
    xv <- vapply(SEEDS, function(s) {
      f <- .need(.cross_file(d, s), paste("cross-dataset result for", st))
      q <- read_csv(f, show_col_types = FALSE)
      q$ap50[grepl("^OVERALL", q$class)][1] * 100
    }, numeric(1))
    out[[length(out) + 1]] <- tibble(stage = st, indomain = im, indomain_sd = isd,
                                     cross = mean(xv), cross_sd = sd(xv),
                                     drop = im - mean(xv))
  }
  bind_rows(out) %>% mutate(stage = factor(stage, levels = STAGE_LEVELS))
}

read_cleantest <- function() {
  out <- list()
  for (m in names(DIR_IN)) {
    d <- DIR_IN[[m]]; v <- c()
    for (s in SEEDS) {
      f1 <- file.path(CT, d, paste0(d, "_seed_", s, "_eval_results.csv"))
      f2 <- file.path(CT, d, paste0("seed_", s), "summary.csv")
      if (file.exists(f1)) v <- c(v, read_csv(f1, show_col_types = FALSE)$mAP_50[1] * 100)
      else if (file.exists(f2)) v <- c(v, read_csv(f2, show_col_types = FALSE)$mAP50[1] * 100)
    }
    if (length(v) == length(SEEDS))
      out[[length(out) + 1]] <- tibble(model = m, clean = mean(v), clean_sd = sd(v))
    else
      warning("leakage-free results incomplete for ", m, " — skipped")
  }
  bind_rows(out)
}

read_training <- function(stages = STAGE_DIR) {
  out <- list()
  for (nm in names(stages)) for (s in SEEDS) {
    g <- Sys.glob(file.path(LOGS, stages[[nm]], paste0("seed_", s), "*history.csv"))
    if (!length(g)) next
    d <- read_csv(g[1], show_col_types = FALSE)
    out[[length(out) + 1]] <- tibble(model = nm, seed = s, epoch = seq_len(nrow(d)),
                                     train_loss = d$train_loss, val_loss = d$val_loss,
                                     map50 = d$map50 * 100, map5095 = d$map50_95 * 100)
  }
  if (!length(out)) stop("no training histories found under ", LOGS)
  bind_rows(out)
}

#' Confusion matrices, seed 42.
#'
#' IMPORTANT. The stored matrix has ten rows. The nine class rows sum to that
#' class's ground-truth count, so normalising by row gives per-class recall.
#' The tenth row, "background", holds false positives grouped by predicted
#' class and sums to the total number of false positives — a different
#' quantity on a different denominator. It is returned separately as
#' `$fp` rather than mixed into the heatmap, where a shared colour scale
#' would make the two look comparable when they are not.
#'
#' Matching protocol (paper_files/confusion_matrix.py): IoU >= 0.5, greedy by
#' best IoU, each ground-truth box claimed once, predictions below a
#' confidence of 0.05 discarded. Note this confidence threshold differs from
#' the 0.01 used for the mAP figures in Table 1.
read_confusion <- function(seed = 42) {
  mats <- list(); fps <- list()
  for (m in names(DIR_X)) {
    f <- .need(file.path(RES, DIR_X[[m]], paste0("seed_", seed),
                         "confusion_matrix_test.csv"),
               paste("confusion matrix for", m))
    d  <- read_csv(f, show_col_types = FALSE)
    gt <- canon_class(d[[1]]); mat <- d[, -1]
    names(mat) <- canon_class(names(mat))
    long <- as_tibble(mat) %>% mutate(true_class = gt) %>%
      pivot_longer(-true_class, names_to = "pred_class", values_to = "count")
    mats[[length(mats) + 1]] <- long %>%
      filter(true_class != "background") %>%
      group_by(true_class) %>% mutate(row_norm = count / sum(count)) %>%
      ungroup() %>% mutate(model = m)
    fps[[length(fps) + 1]] <- long %>%
      filter(true_class == "background", pred_class != "background") %>%
      mutate(model = m, share = count / sum(count)) %>%
      select(model, pred_class, count, share)
  }
  list(cm = bind_rows(mats) %>%
         mutate(model = factor(model, levels = MODEL_LEVELS),
                true_class = factor(true_class, levels = CLASS_LEVELS),
                pred_class = factor(pred_class,
                                    levels = c(CLASS_LEVELS, "background"))),
       fp = bind_rows(fps) %>%
         mutate(model = factor(model, levels = MODEL_LEVELS),
                pred_class = factor(pred_class, levels = CLASS_LEVELS)))
}

#' Detection-level precision, recall and F1, derived from the stored
#' confusion matrices so the numbers are reproducible from this folder.
#'
#' TP = sum of the diagonal over the nine class rows.
#' FN = the background COLUMN: ground-truth boxes that received no detection.
#' FP = the background ROW: detections that matched no ground-truth box.
#'
#' Same matching protocol as read_confusion(): IoU >= 0.50, greedy by best
#' IoU, each ground-truth box claimed once, predictions below a confidence
#' of 0.05 discarded. These are therefore operating-point metrics at that
#' threshold, not threshold-free summaries like mAP — state the threshold
#' wherever they are reported.
read_detection_prf <- function(seed = 42) {
  cm <- read_confusion(seed)$cm
  tp_fn <- cm %>%
    group_by(model) %>%
    summarise(
      tp = sum(count[as.character(true_class) == as.character(pred_class)]),
      fn = sum(count[as.character(pred_class) == "background"]),
      .groups = "drop")
  fp <- read_confusion(seed)$fp %>%
    group_by(model) %>% summarise(fp = sum(count), .groups = "drop")
  tp_fn %>%
    left_join(fp, by = "model") %>%
    mutate(Precision = tp / (tp + fp),
           Recall    = tp / (tp + fn),
           F1        = 2 * Precision * Recall / (Precision + Recall)) %>%
    select(model, tp, fp, fn, Precision, Recall, F1)
}

#' Every annotation in the dataset, read from the raw YOLO label files.
#' All 67,595 boxes — no sampling, so counts and percentages are exact.
read_dataset_boxes <- function() {
  if (!dir.exists(DATA))
    stop("dataset labels not found at ", DATA,
         "\nFig 1 needs them. Other figures do not.")
  out <- list()
  for (sp in c("train", "val", "test")) {
    fs <- list.files(file.path(DATA, sp, "labels"), "\\.txt$", full.names = TRUE)
    m <- do.call(rbind, lapply(fs, function(f) {
      x <- tryCatch(read.table(f), error = function(e) NULL)
      if (is.null(x) || ncol(x) < 5) return(NULL)
      as.matrix(x[, 1:5])
    }))
    if (is.null(m)) next
    out[[sp]] <- tibble(split = sp, class = CLASS_LEVELS[m[, 1] + 1],
                        x_centre = m[, 2], y_centre = m[, 3],
                        width = m[, 4], height = m[, 5],
                        area_frac = m[, 4] * m[, 5],
                        aspect = m[, 4] / m[, 5])
  }
  bind_rows(out) %>%
    mutate(split = factor(split, levels = c("train", "val", "test")),
           class = factor(class, levels = CLASS_LEVELS))
}

read_speed <- function() {
  d <- read_csv(.need(file.path(RES, "fps_benchmark.csv"), "fps_benchmark.csv"),
                show_col_types = FALSE)
  lbl <- c("SSDLite-MobileNetV3-Large" = "SSDLite",
           "YOLOv8n" = "YOLOv8n",
           "RetinaNet-ResNet50-FPN-v2" = "RetinaNet-R50",
           "RetinaNeXt" = "RetinaNeXt", "RetinaNet-Hybrid v3" = "RetinaNeXt")
  d %>% mutate(model = unname(lbl[model]),
               model = factor(model, levels = MODEL_LEVELS))
}

read_pr_curves <- function() {
  f <- .need(file.path(SRC, "pr_curves", "fig7_pr_curves_REAL_values.csv"),
             "measured PR curve points (run make_real_pr_curves.py)")
  read_csv(f, show_col_types = FALSE) %>%
    pivot_longer(-recall, names_to = "class", values_to = "precision") %>%
    mutate(class = factor(canon_class(class), levels = CLASS_LEVELS))
}

message("Setup loaded. Reading CSVs from: ", SRC)
