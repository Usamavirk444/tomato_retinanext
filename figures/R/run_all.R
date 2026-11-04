# ─────────────────────────────────────────────────────────────────────
#  Runs every figure script in order.
#  Open in RStudio and press Source. Nothing to set up first.
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

scripts <- sort(list.files(.here, "^0[1-8].*\\.R$", full.names = TRUE))
failed  <- character(0)

for (f in scripts) {
  message("\n=====================================================")
  message("  ", basename(f))
  message("=====================================================")
  ok <- tryCatch({ source(f, local = new.env()); TRUE },
                 error = function(e) {
                   message("  FAILED: ", conditionMessage(e)); FALSE })
  if (!ok) failed <- c(failed, basename(f))
}

figs <- sort(list.files(file.path(.here, "figures"), "\\.(svg|png|pdf)$"))
message("\n=====================================================")
message("  ", length(scripts) - length(failed), " of ", length(scripts),
        " scripts ran;  ", length(figs), " files in figures/")
if (length(failed)) message("  failed: ", paste(failed, collapse = ", "))
message("=====================================================")
print(figs)
