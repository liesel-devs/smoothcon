#!/usr/bin/env Rscript

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Johannes Brachem
#
# Generate mgcv oracle matrices.
# The adjacent generate.py driver installs the requested mgcv source tree into an
# isolated library, invokes this script, and converts these CSV intermediates to NPZ.

suppressPackageStartupMessages(library(mgcv))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: generate.R OUTPUT_DIRECTORY")
output <- args[[1]]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
writeLines(as.character(packageVersion("mgcv")), file.path(output, "version.txt"))
options(digits = 17)

write_matrix <- function(value, path) {
  write.table(
    data.matrix(value),
    file = path,
    row.names = FALSE,
    col.names = FALSE,
    sep = ","
  )
}

write_case <- function(
  name,
  specification,
  data,
  newdata,
  knots = NULL,
  include_transformed = FALSE
) {
  smooth <- smoothCon(
    specification,
    data = data,
    knots = knots,
    absorb.cons = FALSE,
    diagonal.penalty = FALSE,
    scale.penalty = FALSE
  )[[1]]
  directory <- file.path(output, name)
  dir.create(directory, showWarnings = FALSE)
  write_matrix(data, file.path(directory, "x.csv"))
  write_matrix(newdata, file.path(directory, "new_x.csv"))
  write_matrix(smooth$X, file.path(directory, "basis.csv"))
  write_matrix(smooth$S[[1]], file.path(directory, "penalty.csv"))
  write_matrix(PredictMat(smooth, newdata), file.path(directory, "new_basis.csv"))
  if (include_transformed) {
    transformed <- smoothCon(
      specification,
      data = data,
      knots = knots,
      absorb.cons = TRUE,
      diagonal.penalty = TRUE,
      scale.penalty = TRUE
    )[[1]]
    write_matrix(
      transformed$X,
      file.path(directory, "transformed_basis.csv")
    )
    write_matrix(
      transformed$S[[1]],
      file.path(directory, "transformed_penalty.csv")
    )
    write_matrix(
      PredictMat(transformed, newdata),
      file.path(directory, "transformed_new_basis.csv")
    )
  }
  writeLines(
    c(
      paste("rank", smooth$rank),
      paste("nullity", smooth$null.space.dim)
    ),
    file.path(directory, "metadata.txt")
  )
}

x <- sort(unique(c(seq(-1.8, 2.2, length.out = 31), -0.7, 0.15, 1.41)))
new_x <- c(-2.4, -1.8, -1.1, -0.05, 0.8, 1.75, 2.2, 2.8)
data_1d <- data.frame(x = x)
new_1d <- data.frame(x = new_x)

write_case(
  "ps",
  s(x, bs = "ps", k = 9, m = c(2, 2)),
  data_1d,
  new_1d,
  include_transformed = TRUE
)
write_case("bs", s(x, bs = "bs", k = 9, m = c(3, 2)), data_1d, new_1d)
write_case("cp", s(x, bs = "cp", k = 9, m = c(2, 2)), data_1d, new_1d)
write_case("cr", s(x, bs = "cr", k = 9), data_1d, new_1d)
write_case("cs", s(x, bs = "cs", k = 9), data_1d, new_1d)
write_case("cc", s(x, bs = "cc", k = 9), data_1d, new_1d)
write_case("tp_1d", s(x, bs = "tp", k = 9, m = 2), data_1d, new_1d)

x2 <- seq(-1.5, 1.7, length.out = 36)
y2 <- sin(seq(0.1, 4.9, length.out = 36)) + seq(-0.2, 0.25, length.out = 36)
data_2d <- data.frame(x = x2, y = y2)
new_2d <- data.frame(
  x = c(-1.7, -1.2, -0.3, 0.4, 1.1, 1.9),
  y = c(-1.1, 0.2, 0.95, -0.65, 0.35, 1.2)
)
write_case("tp_2d", s(x, y, bs = "tp", k = 12, m = 2), data_2d, new_2d)
write_case("ts_2d", s(x, y, bs = "ts", k = 12, m = 2), data_2d, new_2d)

write_case("gp_spherical", s(x, y, bs = "gp", k = 12, m = c(1, -1, 1)), data_2d, new_2d)
write_case("gp_power", s(x, y, bs = "gp", k = 12, m = c(2, -1, 1.4)), data_2d, new_2d)
write_case("gp_matern15", s(x, y, bs = "gp", k = 12, m = c(3, -1, 1)), data_2d, new_2d)
write_case("gp_matern25", s(x, y, bs = "gp", k = 12, m = c(4, 2.3, 1)), data_2d, new_2d)
write_case("gp_matern35_stationary", s(x, y, bs = "gp", k = 12, m = c(-5, -1, 1)), data_2d, new_2d)

regions <- factor(
  c("a", "b", "c", "d", "e", "f", "a", "c", "e", "f", "b", "d"),
  levels = c("a", "b", "c", "d", "e", "f")
)
new_regions <- factor(c("f", "a", "d", "c"), levels = levels(regions))
nb <- list(
  a = c("b"),
  b = c("a", "c"),
  c = c("b", "d"),
  d = c("c", "e"),
  e = c("d", "f"),
  f = c("e")
)
write_case(
  "mrf_full",
  s(region, bs = "mrf", k = -1, xt = list(nb = nb)),
  data.frame(region = regions),
  data.frame(region = new_regions)
)
write_case(
  "mrf_low_rank",
  s(region, bs = "mrf", k = 4, xt = list(nb = nb)),
  data.frame(region = regions),
  data.frame(region = new_regions)
)
