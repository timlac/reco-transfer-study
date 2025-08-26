library(lme4)
library(lmerTest)
library(dplyr)

df$condition <- relevel(factor(df$condition), ref = "original")

au_results <- df %>%
  group_by(AU) %>%
  do({
    m <- lmer(activation ~ condition + (1|video_id) + (1|emotion), data = ., REML = FALSE)
    s <- summary(m)$coefficients
    data.frame(
      AU = unique(.$AU),
      intercept = s["(Intercept)", "Estimate"],
      intercept_p = s["(Intercept)", "Pr(>|t|)"],
      furhat_est = s["conditionfurhat", "Estimate"],
      furhat_p   = s["conditionfurhat", "Pr(>|t|)"],
      meta_est   = s["conditionmetahuman", "Estimate"],
      meta_p     = s["conditionmetahuman", "Pr(>|t|)"]
    )
  })

print(au_results)
