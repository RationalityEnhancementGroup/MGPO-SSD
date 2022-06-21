library(nparLD)

# Read data
dat = read.csv("./data/tutor_experiment/tutor_experiment_exclusion_data.csv")
df <- dat[,c("Participant", "Condition", "TrialId", "ExpectedScore", "Score", "ClickAgreement", "RepeatAgreement", "TermAgreement", "GoalStrategy")]

df$Participant <- as.factor(df$Participant)
df$TrialId <- as.factor(df$TrialId)

df$Regret <- dat$BmpsReward - dat$Score

runAnova <- function(df){
  # This one doesn't work for some conditions
  y <- df$TermAgreement
  time <- df$TrialId
  group <- df$Condition
  subject <- df$Participant
  f1 <- f1.ld.f1(y, time, group, subject, time.name = "Trial", group.name = "Condition", description = FALSE, order.warning=FALSE, show.covariance = TRUE) #
  print(f1$RTE)
  print("F1 Walt test")
  print(f1$Wald.test)
  print("F1 Anova test")
  print(f1$ANOVA.test)
  print("Anova box")
  print(f1$ANOVA.test.mod.Box)
  print("F1 pairwise")
  print(f1$pair.comparison)
}

runAnova(df)