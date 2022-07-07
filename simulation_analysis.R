library(emmeans)

# Read data
df = read.csv("./data/simulation_results/simulation_results.csv")
df$Name <- as.factor(df$Name)
df$EnvType <- as.factor(df$EnvType)
df$Cost <- as.factor(df$Cost)

# ANOVA model
model <- aov(ExpectedReward ~ Name * EnvType * Cost, data=df)
summary(model)

# Test main effect of the used algorithm
TukeyHSD(aov(ExpectedReward ~ Name * EnvType * Cost, data=df), "Name")

# Investigate interaction
meansmodel <- emmeans(model, ~ Name | Cost, adjust="none")
contrast(meansmodel, method="dunnett", ref=1, adjust="none")

meansmodel <- emmeans(model, ~ Name | EnvType, adjust="none")
contrast(meansmodel, method="dunnett", ref=1, adjust="none")