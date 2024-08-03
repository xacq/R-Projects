# Instalación y carga de librerías
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("corrplot", quietly = TRUE)) install.packages("corrplot")
if (!requireNamespace("car", quietly = TRUE)) install.packages("car")
if (!requireNamespace("REdaS", quietly = TRUE)) install.packages("REdaS")
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
if (!requireNamespace("e1071", quietly = TRUE)) install.packages("e1071")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")

library(dplyr)
library(corrplot)
library(car)
library(REdaS)
library(caret)
library(e1071)
library(ggplot2)

# Función para liberar memoria
liberar_memoria <- function() {
  invisible(gc())
}

# Leer y preparar los datos
data <- read.csv("bdd2.csv", encoding = "latin1")
liberar_memoria()

# Verificar los nombres de las columnas
print(colnames(data))
liberar_memoria()

# Eliminar la clase "NO IDENTIFICADO" en la columna 'CONDICION_1'
data <- data %>% filter(CONDICION_1 != "NO IDENTIFICADO")
liberar_memoria()

# Convertir las variables de caracteres en factores y optimizar tipos de datos
data <- data %>% mutate_if(is.character, as.factor)
data$ANIO <- as.integer(data$ANIO)
data$SINIESTROS <- as.integer(data$SINIESTROS)
data$LESIONADOS <- as.integer(data$LESIONADOS)
data$FALLECIDOS <- as.integer(data$FALLECIDOS)
liberar_memoria()

# Imprimir la estructura de los datos después de la limpieza
str(data)
liberar_memoria()

# Representación gráfica de los datos
ggplot(data, aes(x = SINIESTROS, y = LESIONADOS)) + 
  geom_point() + 
  labs(title = "Relación entre Siniestros y Lesionados")

# Análisis de correlación
corr_matrix <- cor(data %>% select_if(is.numeric))
liberar_memoria()

# Crear el gráfico de correlación
corrplot(corr_matrix, method = "color", tl.cex = 0.6)

# Detectar y eliminar outliers usando Cook's distance, standardized residuals y rstudent
eliminar_influencias <- function(df, formula) {
  model <- lm(formula, data = df)
  cooksd <- cooks.distance(model)
  influential <- as.numeric(names(cooksd)[(cooksd > (4 / length(cooksd)))])
  df <- df[-influential, ]
  
  studres <- rstudent(model)
  df <- df[abs(studres) < 3, ]
  
  return(df)
}

data <- eliminar_influencias(data, SINIESTROS ~ LESIONADOS + FALLECIDOS + ANIO + ENTE_DE_CONTROL + ZONA + FERIADO)
data <- eliminar_influencias(data, LESIONADOS ~ SINIESTROS + FALLECIDOS + ANIO + ENTE_DE_CONTROL + ZONA + FERIADO)
data <- eliminar_influencias(data, FALLECIDOS ~ SINIESTROS + LESIONADOS + ANIO + ENTE_DE_CONTROL + ZONA + FERIADO)
liberar_memoria()

# Modelo de Clasificación con Regresión Logística
variables_seleccionadas <- c("ANIO", "SINIESTROS", "LESIONADOS", "FALLECIDOS", "ENTE_DE_CONTROL", "ZONA", "FERIADO")
data_class <- data %>% select(variables_seleccionadas, CONDICION_1)
liberar_memoria()

# Dividir manualmente los datos en conjuntos de entrenamiento y prueba
set.seed(123)
trainIndex <- sample(1:nrow(data_class), 0.7 * nrow(data_class))
train_data_class <- data_class[trainIndex, ]
test_data_class <- data_class[-trainIndex, ]
liberar_memoria()

# Construir un modelo de clasificación utilizando Regresión Logística
model_class <- glm(CONDICION_1 ~ ., data = train_data_class, family = binomial)
liberar_memoria()

# Predecir en el conjunto de prueba
predictions_class <- predict(model_class, newdata = test_data_class, type = "response")
liberar_memoria()

# Convertir las predicciones a clases
predictions_class <- ifelse(predictions_class > 0.5, "1", "0")

# Evaluar el modelo de clasificación
conf_matrix <- confusionMatrix(as.factor(predictions_class), test_data_class$CONDICION_1)
print(conf_matrix)
liberar_memoria()

# Modelo de Regresión con Regresión Lineal
variables_seleccionadas_reg <- c("ANIO", "SINIESTROS", "FALLECIDOS", "ENTE_DE_CONTROL", "ZONA", "FERIADO")
data_reg <- data %>% select(all_of(variables_seleccionadas_reg), LESIONADOS)

# Dividir manualmente los datos en conjuntos de entrenamiento y prueba
set.seed(123)
trainIndex_reg <- sample(1:nrow(data_reg), 0.7 * nrow(data_reg))
train_data_reg <- data_reg[trainIndex_reg, ]
test_data_reg <- data_reg[-trainIndex_reg, ]
liberar_memoria()

# Construir un modelo de regresión utilizando Regresión Lineal
model_reg <- lm(LESIONADOS ~ ., data = train_data_reg)
liberar_memoria()

# Predecir en el conjunto de prueba
predictions_reg <- predict(model_reg, newdata = test_data_reg)
liberar_memoria()

# Evaluar el modelo de regresión
mse_reg <- mean((test_data_reg$LESIONADOS - predictions_reg)^2)
print(paste("MSE en el conjunto de prueba:", mse_reg))
liberar_memoria()

# Comprobación de supuestos para la regresión lineal
# Homocedasticidad
plot(model_reg$residuals ~ model_reg$fitted.values)
abline(h = 0, col = "red")

# Normalidad de los residuos
hist(model_reg$residuals)
qqnorm(model_reg$residuals)
qqline(model_reg$residuals, col = "red")

# Reducción de Dimensionalidad con PCA
numeric_cols <- data %>% select_if(is.numeric) %>% names()
data_pca <- data %>% select(all_of(numeric_cols))

# Eliminar columnas constantes
data_pca <- data_pca[, sapply(data_pca, function(x) sd(x) > 0)]

# Escalar los datos
data_scaled <- scale(data_pca)

# Aplicar PCA
pca_result <- prcomp(data_scaled, scale. = TRUE)

# Resumen de PCA
summary(pca_result)

# Seleccionar las componentes principales que expliquen al menos el 90% de la varianza
explained_variance <- summary(pca_result)$importance[3, ]
num_components <- which(cumsum(explained_variance) >= 0.90)[1]

# Crear un nuevo dataframe con las componentes principales seleccionadas
data_pca_selected <- as.data.frame(pca_result$x[, 1:num_components])

# Añadir la variable objetivo 'CONDICION_1' para el modelo de clasificación
data_pca_class <- cbind(data_pca_selected, CONDICION_1 = data$CONDICION_1)

# Añadir la variable objetivo 'LESIONADOS' para el modelo de regresión
data_pca_reg <- cbind(data_pca_selected, LESIONADOS = data$LESIONADOS)
liberar_memoria()

# Validación de los modelos con datos reducidos

# Dividir los datos en conjuntos de entrenamiento y prueba para clasificación con PCA
set.seed(123)
trainIndex_pca_class <- sample(1:nrow(data_pca_class), 0.7 * nrow(data_pca_class))
train_data_pca_class <- data_pca_class[trainIndex_pca_class, ]
test_data_pca_class <- data_pca_class[-trainIndex_pca_class, ]
liberar_memoria()

# Construir y evaluar el modelo de clasificación con datos reducidos
model_pca_class <- glm(CONDICION_1 ~ ., data = train_data_pca_class, family = binomial)
liberar_memoria()

# Verificar que el modelo de clasificación se haya creado correctamente
if (exists("model_pca_class")) {
  # Predecir en el conjunto de prueba
  predictions_pca_class <- predict(model_pca_class, newdata = test_data_pca_class, type = "response")
  predictions_pca_class <- ifelse(predictions_pca_class > 0.5, "1", "0")
  
  # Evaluar el modelo de clasificación
  conf_matrix_pca <- confusionMatrix(as.factor(predictions_pca_class), test_data_pca_class$CONDICION_1)
  print(conf_matrix_pca)
} else {
  print("El modelo de clasificación con PCA no se ha creado correctamente.")
}

# Dividir los datos en conjuntos de entrenamiento y prueba para regresión con PCA
trainIndex_pca_reg <- sample(1:nrow(data_pca_reg), 0.7 * nrow(data_pca_reg))
train_data_pca_reg <- data_pca_reg[trainIndex_pca_reg, ]
test_data_pca_reg <- data_pca_reg[-trainIndex_pca_reg, ]
liberar_memoria()

# Construir y evaluar el modelo de regresión con datos reducidos
model_pca_reg <- lm(LESIONADOS ~ ., data = train_data_pca_reg)
liberar_memoria()

# Verificar que el modelo de regresión se haya creado correctamente
if (exists("model_pca_reg")) {
  # Predecir en el conjunto de prueba
  predictions_pca_reg <- predict(model_pca_reg, newdata = test_data_pca_reg)
  
  # Evaluar el modelo de regresión
  mse_pca_reg <- mean((test_data_pca_reg$LESIONADOS - predictions_pca_reg)^2)
  print(paste("MSE en el conjunto de prueba con PCA:", mse_pca_reg))
} else {
  print("El modelo de regresión con PCA no se ha creado correctamente.")
}


# ***************** ENTRENAMIENTO ********************

# Cargar las librerías necesarias
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
library(dplyr)
library(caret)

# Cargar los datos del archivo
data <- read.csv('ruta_al_archivo/bdd2.csv', encoding = 'latin1')

# Limpieza de datos: eliminar la clase "NO IDENTIFICADO" en la columna 'CONDICION_1'
data <- data %>% filter(CONDICION_1 != "NO IDENTIFICADO")

# Convertir las variables de caracteres en factores y optimizar tipos de datos
data <- data %>% mutate_if(is.character, as.factor)
data$ANIO <- as.integer(data$ANIO)
data$SINIESTROS <- as.integer(data$SINIESTROS)
data$LESIONADOS <- as.integer(data$LESIONADOS)
data$FALLECIDOS <- as.integer(data$FALLECIDOS)

# Seleccionar las variables necesarias para el modelo de clasificación
variables_seleccionadas <- c("ANIO", "SINIESTROS", "LESIONADOS", "FALLECIDOS", "ENTE_DE_CONTROL", "ZONA", "FERIADO")
data_class <- data %>% select(all_of(variables_seleccionadas), CONDICION_1)

# Dividir los datos en conjuntos de entrenamiento y prueba
set.seed(123)  # Para asegurar reproducibilidad
trainIndex <- sample(1:nrow(data_class), 0.7 * nrow(data_class))
train_data_class <- data_class[trainIndex, ]
test_data_class <- data_class[-trainIndex, ]

# Construir un modelo de clasificación utilizando Regresión Logística
model_class <- glm(CONDICION_1 ~ ., data = train_data_class, family = binomial)

# Predecir en el conjunto de prueba
predictions_class <- predict(model_class, newdata = test_data_class, type = "response")

# Convertir las predicciones a clases
predictions_class <- ifelse(predictions_class > 0.5, "1", "0")

# Evaluar el modelo de clasificación
conf_matrix <- confusionMatrix(as.factor(predictions_class), test_data_class$CONDICION_1)

# Mostrar la matriz de confusión y otras métricas
print(conf_matrix)


# ***************************TEST ****************************

# Dividir los datos en conjuntos de entrenamiento y prueba
set.seed(123)  # Para asegurar reproducibilidad
trainIndex <- sample(1:nrow(data_class), 0.7 * nrow(data_class))
train_data_class <- data_class[trainIndex, ]
test_data_class <- data_class[-trainIndex, ]

# Construir un modelo de clasificación utilizando Regresión Logística
model_class <- glm(CONDICION_1 ~ ., data = train_data_class, family = binomial)

# Predecir en el conjunto de prueba
predictions_class <- predict(model_class, newdata = test_data_class, type = "response")

# Convertir las predicciones a clases (1 o 0)
predictions_class <- ifelse(predictions_class > 0.5, "1", "0")

# Evaluar el modelo de clasificación
conf_matrix <- confusionMatrix(as.factor(predictions_class), test_data_class$CONDICION_1)

# Mostrar la matriz de confusión y otras métricas
print(conf_matrix)







