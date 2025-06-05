install.packages("brant")
library(brant)
library(MASS)

# 2. Load data
df <- read.csv("fetal_health.csv")

# Tampilkan 5 baris awal
head(df, 5)

# Struktur data dan tipe variabel
str(df)
install.packages("psych")
library(psych)

install.packages("skimr")
library(skimr)

skim(df)

describe(df)

# Statistik deskriptif
summary(df)

# Distribusi kelas target
table(df$fetal_health)

# Visualisasi distribusi kelas target
ggplot(df, aes(x = factor(fetal_health))) + 
  geom_bar(fill = "steelblue") + 
  labs(title = "Distribusi Kelas Fetal Health", x = "Kelas Fetal Health", y = "Jumlah") +
  theme_minimal()

# --- Preprocessing ---

# 1. Cek missing value
colSums(is.na(df))

# 2. Rename kelas target menjadi faktor dengan label
df$fetal_health <- factor(df$fetal_health,
                          levels = c(1, 2, 3),
                          labels = c("Normal", "Suspect", "Pathological"))

# 3. Cek duplikat data
sum(duplicated(df))

# 4. Hapus duplikat jika ada
df <- df %>% distinct()

# 5. Pisahkan fitur dan target
X <- df %>% select(-fetal_health)
y <- df$fetal_health

# 6. Split data latih dan uji (80:20) dengan stratified sampling
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# 7. Standarisasi fitur (mean=0, sd=1) dengan preProcess caret
preProcValues <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preProcValues, X_train)
X_test_scaled <- predict(preProcValues, X_test)

# 8. Visualisasi boxplot salah satu fitur per kelas target (contoh baseline_value_of_fetal_heart_rate)
df_train <- cbind(X_train_scaled, fetal_health = y_train)

ggplot(df_train, aes(x = fetal_health, y = baseline_value_of_fetal_heart_rate)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Boxplot Baseline FHR berdasarkan Kelas", x = "Kelas Fetal Health", y = "Baseline FHR (Standar)") +
  theme_minimal()


#Uji Asumsi Untuk LDA
data <- read.csv("fetal_health.csv")
data$fetal_health <- as.factor(data$fetal_health)

# Cek struktur data dan missing value
str(data)
print(colSums(is.na(data)))

# Ambil fitur numerik untuk uji Mardia dan Box's M
X <- data %>% select(-fetal_health)
group <- data$fetal_health

# 1. Uji Normalitas Multivariat Mardia
mardia_result <- mvn(data = X, mvnTest = "mardia")
print(mardia_result$multivariateNormality)

# 2. Uji Homogenitas Matriks Kovarians Box's M Test
# cek tabel
table(data_clean$fetal_health)

library(tidyverse)
library(biotools)

X <- data_clean %>% select(-fetal_health)

# PCA dengan scaling
pca <- prcomp(X, scale. = TRUE)

# Pilih jumlah PC yang menjelaskan >90% variansi (misal 10 PC)
var_explained <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
num_pc <- which(var_explained >= 0.9)[1]

X_pca <- as.data.frame(pca$x[, 1:num_pc])
X_pca$fetal_health <- data_clean$fetal_health

# Jalankan Box's M test pada data PCA
boxm_result <- boxM(X_pca %>% select(-fetal_health), X_pca$fetal_health)
print(boxm_result)



#Uji Asumsi Untuk Regresi Logistik Ordinal
#BRANT
data <- read.csv('fetal_health.csv')

model <- polr(factor(fetal_health) ~ accelerations + fetal_movement + uterine_contractions + light_decelerations + severe_decelerations + prolongued_decelerations + abnormal_short_term_variability, data = data)

brant(model)

#VIF
library(MASS)
model <- polr(factor(fetal_health) ~ accelerations + fetal_movement + uterine_contractions + light_decelerations + severe_decelerations + prolongued_decelerations + abnormal_short_term_variability , data = data)

vif(model)


###UJI WALD
# Install dan load package yang diperlukan
install.packages("nnet")
install.packages("car")

library(nnet)
library(car)

data <- read.csv('fetal_health.csv')

sum(is.na(data))

data <- data[, !(names(data) %in% c("Unnamed: 0"))]

target <- data$fetal_health
features <- data[, -which(names(data) == "fetal_health")]

model <- multinom(target ~ ., data = data)

summary(model)

wald.test(b = coef(model), Sigma = vcov(model))

coefficients <- coef(model)
cov_matrix <- vcov(model)

std_errors <- sqrt(diag(cov_matrix))

z_values <- coefficients / std_errors

p_values <- 2 * (1 - pnorm(abs(z_values)))

z_values
p_values


install.packages("nnet")
library(nnet)


###UJI LIKELIHOOD RATIO'
model_kompleks <- multinom(target ~ ., data = data)

model_sederhana <- multinom(target ~ baseline.value + accelerations + fetal_movement, data = data)

log_likelihood_kompleks <- logLik(model_kompleks)
log_likelihood_sederhana <- logLik(model_sederhana)

lr_statistic <- -2 * (log_likelihood_sederhana - log_likelihood_kompleks)

df <- length(coef(model_kompleks)) - length(coef(model_sederhana))

p_value <- 1 - pchisq(lr_statistic, df)

lr_statistic
p_value


####UJI WILKS LAMBDA
selected_vars <- c("baseline.value", "accelerations", "fetal_movement",
                   "uterine_contractions", "light_decelerations", "severe_decelerations")

formula_subset <- as.formula(paste("cbind(", paste(selected_vars, collapse = ","), ") ~ fetal_health"))

manova_result_subset <- manova(formula_subset, data = data)

# Uji Wilks' Lambda
summary(manova_result_subset, test = "Wilks")


###WILKS dengan PCA
pca_data <- prcomp(data[ , -which(names(data) == "fetal_health")], scale. = TRUE)

pca_scores <- as.data.frame(pca_data$x[, 1:5])
pca_scores$fetal_health <- as.factor(data$fetal_health)

manova_pca <- manova(cbind(PC1, PC2, PC3, PC4, PC5) ~ fetal_health, data = pca_scores)

summary(manova_pca, test = "Wilks")


#REVISI
data <- data[ , !(names(data) %in% c("Unnamed..0"))]

data$fetal_health <- as.factor(data$fetal_health)

#hanya ambil 10 prediktor terpilih karena Jumlah observasi dalam setiap kategori fetal_health > jumlah variabel prediktor.
selected_vars <- c("baseline.value", "accelerations", "fetal_movement",
                   "uterine_contractions", "light_decelerations",
                   "abnormal_short_term_variability", "mean_value_of_short_term_variability",
                   "histogram_mean", "histogram_variance", "histogram_tendency")

formula <- as.formula(paste("cbind(", paste(selected_vars, collapse = ", "), ") ~ fetal_health"))
manova_result <- manova(formula, data = data)
#wilks
summary(manova_result, test = "Wilks")

#hasil uji parsial
summary.aov(manova_result)

#hasil uji parsial menggunakan wald test
# Loop untuk setiap variabel respon dan lakukan Wald test untuk pengaruh fetal_health
for (var in selected_vars) {
  formula <- as.formula(paste(var, "~ fetal_health"))
  
  model <- lm(formula, data = data)
  
  cat("Wald test (Type III) untuk variabel:", var, "\n")
  
  #lakukan uji Anova dengan Type III mirip Wald test
  print(Anova(model, type = 3))
  
  cat("\n--------------------------------------------\n\n")
}


####LDA
data <- read.csv('fetal_health.csv')

data$fetal_health <- as.factor(data$fetal_health)

str(data)

grouped_data <- split(data, data$fetal_health)

constant_vars <- sapply(grouped_data, function(group) {
  apply(group[, -which(names(group) == "fetal_health")], 2, function(col) length(unique(col)) == 1)
})

constant_vars

data_filtered <- data[, !names(data) %in% c("severe_decelerations")]

str(data_filtered)

library(MASS)

lda_model <- lda(fetal_health ~ ., data = data_filtered)

print(lda_model)

lda_pred <- predict(lda_model, newdata = data_filtered)

head(lda_pred$class)

accuracy <- sum(lda_pred$class == data_filtered$fetal_health) / nrow(data_filtered)
print(paste("Akurasi Model LDA: ", accuracy))

conf_matrix <- table(Predicted = lda_pred$class, Actual = data_filtered$fetal_health)

print(conf_matrix)

#accuracy dari confusion matrix
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy: ", accuracy))

class_names <- rownames(conf_matrix)
for (class in class_names) {
  
  TP <- conf_matrix[class, class]
  FP <- sum(conf_matrix[, class]) - TP
  FN <- sum(conf_matrix[class, ]) - TP
  TN <- sum(conf_matrix) - TP - FP - FN
  
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision <- TP / (TP + FP)
  
  cat("\nFor class:", class, "\n")
  cat("Sensitivity: ", sensitivity, "\n")
  cat("Specificity: ", specificity, "\n")
  cat("Precision: ", precision, "\n")
}

# --- Logistic Regression Ordinal ---
data <- read.csv("fetal_health.csv")

# Membuat model regresi logistik ordinal dengan polr
model <- polr(factor(fetal_health) ~ accelerations + fetal_movement + uterine_contractions + 
                light_decelerations + severe_decelerations + prolongued_decelerations + 
                abnormal_short_term_variability, data = data, Hess=TRUE)

# Ringkasan model untuk melihat koefisien dan standar error
summary(model)

# Menghitung Log Odds Ratio (LOR) dan Confidence Interval
coef_table <- coef(summary(model))
LOR <- coef_table[, "Value"]
SE <- coef_table[, "Std. Error"]
CI_lower <- LOR - 1.96 * SE
CI_upper <- LOR + 1.96 * SE
result <- data.frame(LOR = LOR, Lower95CI = CI_lower, Upper95CI = CI_upper)
print(result)

# Uji Brant untuk menguji asumsi proportional odds
brant_test <- brant(model)
print(brant_test)

# Menghitung Variance Inflation Factor (VIF) untuk memeriksa multikolinearitas
# Perlu membuat model linear dengan formula sama untuk fungsi vif
# Karena vif tidak langsung bisa dipakai pada polr
lm_model <- lm(as.numeric(factor(fetal_health)) ~ accelerations + fetal_movement + uterine_contractions + 
                 light_decelerations + severe_decelerations + prolongued_decelerations + 
                 abnormal_short_term_variability, data = data)
vif_values <- vif(lm_model)
print(vif_values)


# Membuat confusion matrix dengan caret
conf_matrix <- confusionMatrix(predicted, factor(data$fetal_health))

# Mengambil tabel confusion matrix
cm_table <- conf_matrix$table

# Memberi nama baris dan kolom agar jelas
rownames(cm_table) <- paste("Predicted", rownames(cm_table))
colnames(cm_table) <- paste("Actual", colnames(cm_table))

# Menampilkan confusion matrix dengan label jelas
print(cm_table)

# Menampilkan ringkasan evaluasi model lengkap
print(conf_matrix)

# Install dan muat ggplot2 jika belum
install.packages("ggplot2")
library(ggplot2)

# Ubah confusion matrix menjadi data frame
cm_df <- as.data.frame(conf_matrix$table)

# Plot menggunakan ggplot2
ggplot(data = cm_df, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  geom_text(aes(label = Freq), vjust = 0.5, size = 6) +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(title = "Confusion Matrix",
       x = "Actual Class",
       y = "Predicted Class") +
  theme_minimal(base_size = 15)