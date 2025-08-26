library(tidyverse)

df <- read_csv("data/out/openface_data_long_all.csv", show_col_types = FALSE) |>
  filter(condition %in% c("original","furhat","metahuman"))

# Collapse duplicates first
df_agg <- df |>
  summarise(activation = mean(activation, na.rm = TRUE),
            .by = c(video_id, emotion, feature, condition))

# Wide
df_wide <- df_agg |>
  pivot_wider(names_from = condition, values_from = activation)

safe_cor_test <- function(x, y, method = "pearson") {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(tibble(r = NA_real_, p = NA_real_))
  x <- x[ok]; y <- y[ok]
  if (sd(x) == 0 || sd(y) == 0) return(tibble(r = NA_real_, p = NA_real_))
  ct <- cor.test(x, y, method = method)
  tibble(r = unname(ct$estimate), p = ct$p.value)
}

corrs <- df_wide |>
  group_by(feature) |>
  summarise(
    {
      cf <- safe_cor_test(original, furhat)
      cm <- safe_cor_test(original, metahuman)
      fm <- safe_cor_test(furhat, metahuman)
      tibble(
        corr_original_furhat = cf$r, p_orig_furhat = cf$p,
        corr_original_metahuman = cm$r, p_orig_metahuman = cm$p,
        corr_furhat_metahuman = fm$r, p_furhat_metahuman = fm$p
      )
    },
    .groups = "drop"
  )

print(corrs, n = Inf)