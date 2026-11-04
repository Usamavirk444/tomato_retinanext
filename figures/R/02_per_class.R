# ─────────────────────────────────────────────────────────────────────
#  Figure 5 — per-class AP@0.5, nine classes x four models
#
#  Source: source_csv/results/<model>/seed_<n>/per_class_test.csv
#
#  Two layouts are produced from the same data. Set LAYOUT below.
#
#    "bar"       horizontal grouped bars, zero baseline, error bars.
#                Values are read against a labelled axis, so the encoding
#                is undistorted. The safe choice for the paper.
#
#    "circular"  polar bars in the style of the R Graph Gallery circular
#                barplot: one coloured group per model, nine bars per
#                group, the full class name and its AP radiating from
#                each bar tip,
#                a thin coloured arc inside the ring spanning each group,
#                and the model key in the top-right corner.
#
#  WHY THE CIRCULAR VERSION CARRIES A WARNING. AP here spans roughly 45%
#  to 100%. A polar barplot with a zero baseline makes all 36 bars nearly
#  the same length and says nothing, so bars must start part-way out
#  (CIRC_BASE). That truncation corrupts the channel the reader uses
#  first — length. At CIRC_BASE = 40, Healthy at 45.72 against 81.75 is a
#  1.8x difference drawn as a 7.3x difference in bar length, while Late
#  Blight at 96.66 against 98.63 is drawn honestly. The wedge shape adds
#  to it: bars are wider at the outer end, so longer bars carry more ink.
#  Use "circular" for a poster or graphical abstract; use "bar" wherever a
#  reader will extract values.
#
#  Classes are ordered by RetinaNeXt's AP in both layouts.
# ─────────────────────────────────────────────────────────────────────
# Portable resolution of this script's own directory.
# sys.frame(1)$ofile is set when the file is source()d (RStudio "Source",
# run_all.R) but NOT under `Rscript file.R`. The previous fallback was an
# absolute path into tomato_vm_clean/, the read-only research archive, which
# does not exist on other machines and which this script must never write to.
.script_dir <- function() {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
}
.here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)),
                  error = function(e) .script_dir())
source(file.path(.here, "00_setup.R"))

LAYOUT    <- "circular"   # "circular" or "bar"
CIRC_BASE <- 40      # radial start for the circular layout, in AP %
CIRC_GAP  <- 3       # empty slots between model groups
CIRC_RINGS <- c(60)             # full circles drawn as reference rings
CIRC_SCALE <- c(60, 80, 100)    # scale numbers shown in the empty wedge
CIRC_SCALE_AFTER <- 1           # which gap holds them: 1 = after group 1

pc <- read_per_class()

summ <- pc %>%
  group_by(model, class) %>%
  summarise(mean = mean(ap50), sd = sd(ap50), .groups = "drop")

ord  <- summ %>% filter(model == "RetinaNeXt") %>% arrange(mean) %>% pull(class)
summ <- summ %>%
  mutate(class = factor(as.character(class), levels = as.character(ord)))

# ─────────────────────────────────────────────────────────────────────
if (LAYOUT == "bar") {

  p <- ggplot(summ, aes(class, mean, fill = model)) +
    geom_col(position = position_dodge(0.78), width = 0.7) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
                  position = position_dodge(0.78), width = 0.25,
                  linewidth = 0.3, colour = PAL[["grey_dark"]]) +
    coord_flip() +
    scale_fill_manual(values = MODEL_COLS,
                      guide = guide_legend(nrow = 2, reverse = TRUE)) +
    scale_y_metric() +
    labs(subtitle = "Three-seed mean ± standard deviation. Classes ordered by RetinaNeXt AP.",
         x = NULL, y = "AP@0.5 (%)") +
    theme_paper() +
    theme(panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(colour = PAL[["grey_light"]],
                                            linewidth = 0.3))

  save_fig(p, "Fig08_per_class_ap", 7.2, 5.4)

# ─────────────────────────────────────────────────────────────────────
} else if (LAYOUT == "circular") {

  mods <- levels(summ$model)
  ncl  <- nlevels(summ$class)
  blk  <- ncl + CIRC_GAP                       # slots per model group

  # Lay all bars on one linear index, best class first within each group,
  # leaving CIRC_GAP empty slots between groups. coord_polar wraps it.
  d <- summ %>%
    mutate(ci = ncl - as.integer(class) + 1L,  # best AP first
           mi = as.integer(model),
           x  = (mi - 1L) * blk + ci,
           lab = paste0(as.character(class), "  ", sprintf("%.1f", mean)),
           r   = mean - CIRC_BASE)

  ntot <- max(d$x) + CIRC_GAP                  # total slots incl. last gap

  # Gallery label recipe: rotate to the radius, flip on the left half so
  # nothing reads upside-down.
  d <- d %>%
    mutate(angle = 90 - 360 * (x - 0.5) / ntot,
           hjust = ifelse(angle < -90, 1, 0),
           angle = ifelse(angle < -90, angle + 180, angle))

  grp <- d %>% group_by(model) %>%
    summarise(start = min(x) - 0.45, end = max(x) + 0.45, .groups = "drop")

  # Which reference rings to draw as full circles. The 80 and 100 rings
  # ran straight through the bars and the tip labels and added little, so
  # only the inner one is drawn. The scale still reads because every
  # value is printed at its own bar tip.
  rings <- CIRC_RINGS

  # The scale numbers sit in an empty wedge, where nothing else is drawn.
  # CIRC_SCALE_AFTER picks which gap: 1 = the gap after the first group,
  # 2 = after the second, and so on; 0 = the gap before the first.
  scale_x <- if (CIRC_SCALE_AFTER == 0) max(d$x) + CIRC_GAP / 2
             else CIRC_SCALE_AFTER * blk - CIRC_GAP / 2 + 0.5
  scale_lab <- tibble(x = scale_x,
                      y = CIRC_SCALE - CIRC_BASE,
                      label = paste0(CIRC_SCALE, "%"))

  p <- ggplot(d) +
    # radial reference ring(s), under the bars
    geom_hline(yintercept = rings - CIRC_BASE,
               colour = PAL[["grey_light"]], linewidth = 0.3) +
    geom_col(aes(x, r, fill = model), width = 0.86,
             colour = "white", linewidth = 0.25) +
    # tip labels: full class name, radiating outward as in the reference
    geom_text(aes(x, r + 2, label = lab, angle = angle, hjust = hjust),
              size = 1.95, colour = PAL[["grey_dark"]]) +
    # thin arc inside the ring spanning each model group, as in the
    # reference. The reference letters each arc (A/B/C/D); here the
    # legend names the models instead, so the arcs carry colour only.
    geom_segment(data = grp,
                 aes(x = start, xend = end,
                     y = -3.5, yend = -3.5, colour = model),
                 linewidth = 1.0, lineend = "butt", show.legend = FALSE) +
    # scale numbers, stacked in the empty wedge between the last group
    # and the first so they sit clear of every bar and tip label
    geom_text(data = scale_lab, aes(x, y, label = label),
              size = 2.1, colour = PAL[["grey_mid"]]) +
    coord_polar(start = 0) +
    scale_fill_manual(values = MODEL_COLS, name = NULL) +
    scale_colour_manual(values = MODEL_COLS, guide = "none") +
    scale_y_continuous(limits = c(-23, 118 - CIRC_BASE)) +
    labs(subtitle = paste0("AP@0.5, three-seed mean. Bars start at ",
                           CIRC_BASE, "% AP, so bar length is not ",
                           "proportional to AP."),
         x = NULL, y = NULL) +
    theme_paper() +
    theme(axis.text   = element_blank(),
          axis.ticks  = element_blank(),
          panel.grid  = element_blank(),
          # theme_paper() draws axis lines; under coord_polar the x axis
          # line becomes a full circle round the outside of the plot and
          # the y axis line a second one. Both have to go.
          axis.line   = element_blank(),
          panel.border = element_blank(),
          # model key in the top-right corner, inside the panel
          legend.position      = c(0.99, 0.99),
          legend.justification = c(1, 1),
          legend.direction     = "vertical",
          legend.background    = element_rect(fill = "white",
                                              colour = NA),
          legend.key.size      = unit(9, "pt"),
          legend.text          = element_text(size = 6.8),
          plot.margin          = margin(2, 2, 2, 2))

  save_fig(p, "Fig08_per_class_ap_circular", 7.4, 7.8)

} else {
  stop('LAYOUT must be "bar" or "circular"')
}

print(p)
