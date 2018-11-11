# Scripts and tools for analyzing attention maps/DrQA model performance.

library(tidyverse)
library(gplots)
library(cowplot)
library(RColorBrewer)
library(jsonlite)

ATTN_COL <- colorRampPalette(c('white', 'blue'))(1000)

read_attn <- function(expt, n = 0, type = 'q_dialog_attn') {
  csv_fp <- paste0('../exp/', expt, '/attention/', n, '-', type, '.csv')
  message("Reading ", csv_fp)
  read_csv(csv_fp)
}

retrieve_q <- function(attn, n = 1, max_history = -1) {
  if (max_history > -1) {
    past_times_re <- (n - max_history):(n - 1) %>%
      sapply(function(t) paste0('^', t, '-')) %>%
      paste(collapse = '|')
  } else {
    past_times_re <- 0:(n - 1) %>%
      sapply(function(t) paste0('^', t, '-')) %>%
      paste(collapse = '|')
  }
  attn_filtered <- attn %>%
    filter(startsWith(`<QUESTION>`, paste0(n, '-')))
  attn_filtered <- attn_filtered %>%
    select(`<QUESTION>`, matches(past_times_re), contains('<KEEP>'), contains('<MERGE>'))
  attn_filtered
}

attn_hm <- function(attn, n = 1, max_history = -1, hm_title = "") {
  if (n < 1) {
    stop(paste0("Can't draw attention heatmap for n = ", n))
  }
  
  aq <- retrieve_q(attn, n, max_history)
  aq_m <- aq %>%
    select(-`<QUESTION>`) %>%
    as.matrix
  
  rownames(aq_m) <- paste0(aq$`<QUESTION>`, '-k', round(aq$`<KEEP>`, 3))
  
  # Add by one since the math here is 1-indexed
  np <- n + 1
  
  last_col <- length(colnames(aq_m)) - ('<KEEP>' %in% colnames(aq_m)) - ('<MERGE>' %in% colnames(aq_m))
  if (last_col == length(colnames(aq_m))) {
    last_col <- NULL
  }
  
  par(las = 1)
  heatmap.2(aq_m, Rowv = FALSE, Colv = FALSE, dendrogram = 'none',
            col = ATTN_COL, trace = 'none', cexRow = 0.9, cexCol = 0.9,
            key.xtickfun = function() list(at = seq(0, 1, 0.25), labels = seq(0, 1, 0.25)),
            key.ylab = NA,
            key.xlab = NA,
            key.title = NA,
            density.info = 'none',
            keysize = 1,
            key.ytickfun = function() list(at = c(), labels = c()),
            margins = c(7, 7),
            xlab = bquote(bold(d^{.(np)+phantom(0)})),
            ylab = bquote(bold(q^{.(np)})),
            colsep = last_col,
            sepcolor = 'black',
            sepwidth = c(0.1, 0.1),
            breaks = seq(0, 1, length.out = 1001)
            )
  title(hm_title)
}

read_metrics <- function(mns) {
  ms <- names(mns)
  all_metrics <- data.frame(
    epoch = NULL,
    value = NULL,
    metric = NULL,
    model = NULL
  )
  for (m in ms) {
    dir <- paste0('../exp/', m, '/metrics/')
    dev_em_fp <- paste0(dir, 'dev_em_epoch.txt')
    dev_f1_fp <- paste0(dir, 'dev_f1_epoch.txt')
    dev_em <- scan(dev_em_fp, sep='\n')
    dev_f1 <- scan(dev_f1_fp, sep='\n')
    epoch <- seq_along(dev_f1)
    metrics <- data.frame (
      model = mns[m],
      value = dev_f1,
      metric = 'dev_f1',
      epoch = epoch
    )
    all_metrics = rbind(all_metrics, metrics)
  }
  all_metrics
}

read_predictions <- function(expt) {
  csv_fp <- paste0('../exp/', expt, '/predictions.csv')
  message("Reading ", csv_fp)
  read_csv(csv_fp)
}

read_predictions_json <- function(expt) {
  json_fp <- paste0('../exp/', expt, '/predictions.json')
  message("Reading ", json_fp)
  fromJSON(json_fp)
}

read_coqa <- function(fp) {
  data <- fromJSON(fp)$data
  data$id <- 0:(length(data$id) - 1)
  questions <- data$questions
  questions <- lapply(1:length(questions), function(i) {
    df <- questions[[i]]
    df$id <- i - 1
    df$next_question <- c(df$input_text[2:length(df$input_text)], 'NULL')
    df
  })
  questions_df <- do.call(rbind, questions) %>%
    rename(question = input_text)
  
  answers <- data$answers
  answers <- lapply(1:length(answers), function(i) {
    df <- answers[[i]]
    df$id <- i - 1
    df$next_answer <- c('NULL', df$input_text)[1:length(df$input_text)]
    df
  })
  answers_df <- do.call(rbind, answers) %>%
    select(id, turn_id, span_start, span_end, answer = input_text, next_answer)
  
  combined_df <- questions_df %>%
    left_join(answers_df, by = c('id', 'turn_id')) %>%
    group_by(id) %>%
    mutate(history = Reduce(paste, paste0('**\\<Q', turn_id, '\\>** ', question, ' **\\<A', turn_id, '\\>** ', answer, '\n\n'), accumulate = TRUE))
  
  combined_df %>%
    left_join(data %>% select(source, id, filename, story), by = 'id') %>%
    tbl_df
}
