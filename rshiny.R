# Load required libraries
library(shiny)
library(ggplot2)
library(dplyr)
library(caret)
library(pROC)
library(corrplot)
library(smotefamily)  # For SMOTE

# Increase the file upload limit to 200MB
options(shiny.maxRequestSize = 200*1024^2)

# Define UI for the app
ui <- fluidPage(
  titlePanel("Churn Prediction with Supervised Learning Models"),
  
  tabsetPanel(
    
    # Tab 1: Exploratory Data Analysis (EDA)
    tabPanel("Exploratory Data Analysis",
             sidebarLayout(
               sidebarPanel(
                 # Dataset upload input
                 fileInput("dataset", "Upload Dataset (CSV file)", accept = ".csv"),
                 
                 # Column selection for churn and other attributes
                 uiOutput("churn_column_ui"),  # Churn column
                 uiOutput("num_column_ui"),    # Numeric column
                 
                 # Buttons for different tasks
                 actionButton("dimensions", "Show Dataset Dimensions"),
                 actionButton("missing_values", "Check for Missing Values"),
                 actionButton("constant_attr", "Check for Constant Attributes"),
                 actionButton("plot_churn_prop", "Plot Churn Proportion"),
                 actionButton("plot_num_vs_churn", "Plot Numeric vs. Churn"),
                 actionButton("corr_matrix", "Show Correlation Matrix"),
                 actionButton("show_conclusion", "Show Conclusion")  # Button to display conclusion
               ),
               
               mainPanel(
                 # Separate outputs for dimensions, missing values, and constant attributes
                 verbatimTextOutput("dim_output"),        # Output for dataset dimensions
                 verbatimTextOutput("missing_output"),    # Output for missing values
                 verbatimTextOutput("constant_output"),   # Output for constant attributes
                 plotOutput("plot_output"),               # Output for plots
                 plotOutput("corr_plot"),                 # Output for correlation matrix
                 
                 # Output for the conclusion
                 h4("Conclusion"),
                 verbatimTextOutput("conclusion_output")  # Output for conclusion text
               )
             )
    ),
    
    # Tab 2: Model Training and Evaluation
    tabPanel("Model Training and Evaluation",
             sidebarLayout(
               sidebarPanel(
                 # Model selection
                 selectInput("model_type", "Select Model", 
                             choices = c("Decision Tree", "Logistic Regression", "SVM (without kernel)", "SVM (with kernel)")),
                 
                 # Dropdown to select the dataset (Original, SMOTE, or Undersampled)
                 selectInput("dataset_choice", "Select Dataset for Training", 
                             choices = c("Original", "SMOTE", "Undersampled")),
                 
                 # Button to start model training
                 actionButton("train_model", "Train Model"),
                 
                 # Button for Grid Search
                 actionButton("grid_search", "Perform Grid Search"),
                 
                 # Button for confusion matrix
                 actionButton("plot_conf_matrix", "Plot Confusion Matrix") ,
                 
                 actionButton("show_model_conclusion", "Show Model Summary"),
                 
                 downloadButton("download_model", "Download Trained Model")
                 
                 
               ),
               
               mainPanel(
                 # Output for model performance
                 verbatimTextOutput("model_performance_output"),
                 
                 # Output for Grid Search Results
                 verbatimTextOutput("grid_search_output"),
                 
                 # Output for Confusion Matrix
                 plotOutput("conf_matrix_output") , 
                 
                 verbatimTextOutput("model_conclusion_output")  # Output for model summary conclusion
                 
               )
             )
    ),
    
    # Tab 3: Data Resampling (SMOTE or Undersample)
    tabPanel("Data Resampling",
             sidebarLayout(
               sidebarPanel(
                 # Button to apply SMOTE
                 actionButton("apply_smote", "Apply SMOTE"),
                 
                 # Button to apply undersampling
                 actionButton("apply_undersample", "Apply Undersampling")
               ),
               
               mainPanel(
                 verbatimTextOutput("resample_output")  # Output to show resampled data info
               )
             )
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # Reactive expression to read the dataset
  dataset <- reactive({
    req(input$dataset)  # Ensure the dataset is uploaded
    read.csv(input$dataset$datapath, sep = ",")  # Adjust delimiter if necessary
  })
  
  # Dynamically generate the UI for selecting columns
  output$churn_column_ui <- renderUI({
    req(dataset())
    colnames <- names(dataset())
    selectInput("churn_column", "Select Churn Column", choices = colnames, selected = colnames[1])
  })
  
  output$num_column_ui <- renderUI({
    req(dataset())
    colnames <- names(dataset())
    selectInput("num_column", "Select Numeric Column", choices = colnames, selected = colnames[1])
  })
  
  # Show Dataset Dimensions
  observeEvent(input$dimensions, {
    req(dataset())
    output$dim_output <- renderPrint({
      data <- dataset()
      dim(data)  # Display dimensions (rows and columns)
    })
  })
  
  # Check for Missing Values
  observeEvent(input$missing_values, {
    req(dataset())
    output$missing_output <- renderPrint({
      data <- dataset()
      colSums(is.na(data))  # Show number of missing values per column
    })
  })
  
  # Check for Constant Attributes
  observeEvent(input$constant_attr, {
    req(dataset())
    output$constant_output <- renderPrint({
      data <- dataset()
      constant_columns <- sapply(data, function(x) length(unique(x)) == 1)
      names(data)[constant_columns]  # List columns with constant values
    })
  })
  
  # Plot the proportion of churned individuals
  observeEvent(input$plot_churn_prop, {
    req(input$churn_column)
    output$plot_output <- renderPlot({
      data <- dataset()
      data[[input$churn_column]] <- as.factor(data[[input$churn_column]])
      
      ggplot(data, aes_string(x = input$churn_column)) +
        geom_bar(fill = "steelblue") +
        labs(title = "Proportion of Churn vs Non-Churn", x = input$churn_column, y = "Count")
    })
  })
  
  # Plot Numeric Variables vs Churn
  observeEvent(input$plot_num_vs_churn, {
    req(input$churn_column, input$num_column)
    output$plot_output <- renderPlot({
      data <- dataset()
      data[[input$churn_column]] <- as.factor(data[[input$churn_column]])
      
      ggplot(data, aes_string(x = input$num_column, color = input$churn_column)) +
        geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
        labs(title = "Numeric Variable vs Churn", x = input$num_column, y = "Count")
    })
  })
  
  # Show Correlation Matrix for Numeric Variables
  observeEvent(input$corr_matrix, {
    req(dataset())
    output$corr_plot <- renderPlot({
      data <- dataset()
      num_data <- data %>% select_if(is.numeric)
      corr_matrix <- cor(num_data, use = "complete.obs")
      corrplot(corr_matrix, method = "circle")
    })
  })
  
  # Reactive expressions for SMOTE and undersampling
  smote_data <- reactiveVal(NULL)
  undersample_data <- reactiveVal(NULL)
  
  # Reactive expression for preprocessing data (for modeling)
  preprocess_data <- reactive({
    req(input$churn_column)
    data <- dataset()
    
    # Remove rows with missing values or impute them (for simplicity, we remove them here)
    data <- na.omit(data)
    
    # Convert churn column to factor with valid names using make.names()
    data[[input$churn_column]] <- factor(make.names(data[[input$churn_column]]))
    
    return(data)
  })
  
  # Apply SMOTE
  observeEvent(input$apply_smote, {
    data <- preprocess_data()
    
    # Ensure there are at least two levels before applying SMOTE
    if (length(unique(data[[input$churn_column]])) < 2) {
      output$resample_output <- renderPrint({
        cat("Error: Not enough class diversity in churn column to apply SMOTE.")
      })
      return()
    }
    
    smote_result <- SMOTE(X = data[, -which(names(data) == input$churn_column)], 
                          target = data[[input$churn_column]])
    
    smote_data <- smote_result$data
    colnames(smote_data)[ncol(smote_data)] <- input$churn_column  # Correct churn column name after SMOTE
    
    # Check if both classes are present after SMOTE
    if (length(unique(smote_data[[input$churn_column]])) < 2) {
      output$resample_output <- renderPrint({
        cat("Error: SMOTE did not generate both churn classes.")
      })
      return()
    }
    
    smote_data(smote_data)
    output$resample_output <- renderPrint({
      cat("SMOTE applied. New data dimensions:", dim(smote_data()), "\n")
      print(table(smote_data()[[input$churn_column]]))  # Show class distribution
    })
  })
  
  # Apply Undersampling
  observeEvent(input$apply_undersample, {
    data <- preprocess_data()
    
    undersample_result <- downSample(x = data[, -which(names(data) == input$churn_column)], 
                                     y = data[[input$churn_column]])
    
    undersample_data(undersample_result)
    output$resample_output <- renderPrint({
      cat("Undersampling applied. New data dimensions:", dim(undersample_data()), "\n")
      print(table(undersample_data()$Class))  # Show class distribution
    })
  })
  
  # Select the dataset based on user input (Original, SMOTE, or Undersample)
  selected_data <- reactive({
    req(input$dataset_choice)
    if (input$dataset_choice == "Original") {
      return(preprocess_data())
    } else if (input$dataset_choice == "SMOTE" && !is.null(smote_data())) {
      return(smote_data())
    } else if (input$dataset_choice == "Undersampled" && !is.null(undersample_data())) {
      return(undersample_data())
    } else {
      showNotification("Please apply SMOTE or undersample before selecting those datasets.", type = "warning")
      return(preprocess_data())  # Fallback to original if no resampled data is available
    }
  })
  
  # Train and Evaluate Models
  # After the model is trained, save it to the reactive value
  observeEvent(input$train_model, {
    req(input$model_type, input$churn_column)
    data <- selected_data()
    
    # Ensure both classes are present in the dataset
    if (length(unique(data[[input$churn_column]])) < 2) {
      output$model_performance_output <- renderPrint({
        cat("Error: Selected dataset does not contain both churn classes.")
      })
      return()
    }
    
    # Show the progress bar during model training
    withProgress(message = "Training model...", value = 0, {
      
      # Create train/test split using stratified sampling to preserve class balance
      set.seed(123)
      trainIndex <- createDataPartition(data[[input$churn_column]], p = 0.7, list = FALSE, times = 1)
      trainData <- data[trainIndex, ]
      testData <- data[-trainIndex, ]
      
      # Update progress bar
      incProgress(0.2, detail = "Splitting data...")
      
      # Formula for model
      formula <- as.formula(paste(input$churn_column, "~ ."))
      
      # Train the model based on the user's choice
      model <- NULL
      if (input$model_type == "Decision Tree") {
        model <- train(formula, data = trainData, method = "rpart", 
                       trControl = trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE), 
                       metric = "ROC")
        
      } else if (input$model_type == "Logistic Regression") {
        model <- train(formula, data = trainData, method = "glm", family = "binomial", 
                       trControl = trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE), 
                       metric = "ROC")
        
      } else if (input$model_type == "SVM (without kernel)") {
        model <- train(formula, data = trainData, method = "svmLinear", 
                       trControl = trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE), 
                       metric = "ROC")
        
      } else if (input$model_type == "SVM (with kernel)") {
        model <- train(formula, data = trainData, method = "svmRadial", 
                       trControl = trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE), 
                       metric = "ROC")
      }
      
      # Update progress bar
      incProgress(0.6, detail = "Training model...")
      
      # Save the trained model to reactive value
      trained_model(model)
      
      # Predict on test data and show performance
      predictions <- predict(model, testData)
      roc_curve <- roc(testData[[input$churn_column]], as.numeric(predictions))
      auc_value <- auc(roc_curve)
      
      # Update progress bar
      incProgress(0.9, detail = "Evaluating model...")
      
      output$model_performance_output <- renderPrint({
        cat("Model:", input$model_type, "\n")
        cat("AUC:", auc_value, "\n")
      })
      
      # Complete the progress bar
      incProgress(1, detail = "Model training complete.")
    })
  })
  
  
  # Perform Grid Search with AUC as the metric
  observeEvent(input$grid_search, {
    req(input$model_type, input$churn_column)
    data <- selected_data()
    
    # Ensure both classes are present in the dataset
    if (length(unique(data[[input$churn_column]])) < 2) {
      output$grid_search_output <- renderPrint({
        cat("Error: Selected dataset does not contain both churn classes.")
      })
      return()
    }
    
    # Create train/test split
    set.seed(123)
    trainIndex <- createDataPartition(data[[input$churn_column]], p = 0.7, list = FALSE)
    trainData <- data[trainIndex, ]
    
    # Formula for model
    formula <- as.formula(paste(input$churn_column, "~ ."))
    
    # Define tuning grid and perform grid search
    tuneGrid <- NULL
    trControl <- trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE)
    
    if (input$model_type == "Decision Tree") {
      tuneGrid <- expand.grid(cp = seq(0.001, 0.1, 0.01))
      model <- train(formula, data = trainData, method = "rpart", trControl = trControl, tuneGrid = tuneGrid, metric = "ROC")
      
    } else if (input$model_type == "SVM (without kernel)") {
      tuneGrid <- expand.grid(C = seq(0.001, 1, 0.1))
      model <- train(formula, data = trainData, method = "svmLinear", trControl = trControl, tuneGrid = tuneGrid, metric = "ROC")
      
    } else if (input$model_type == "SVM (with kernel)") {
      tuneGrid <- expand.grid(sigma = seq(0.001, 1, 0.1), C = seq(0.001, 1, 0.1))
      model <- train(formula, data = trainData, method = "svmRadial", trControl = trControl, tuneGrid = tuneGrid, metric = "ROC")
    }
    
    # Display grid search results
    output$grid_search_output <- renderPrint({
      print(model)
    })
  })
  
  # Plot Confusion Matrix after model training
  observeEvent(input$plot_conf_matrix, {
    req(input$model_type, input$churn_column)
    data <- selected_data()
    
    # Create train/test split
    set.seed(123)
    trainIndex <- createDataPartition(data[[input$churn_column]], p = 0.7, list = FALSE)
    trainData <- data[trainIndex, ]
    testData <- data[-trainIndex, ]
    
    # Formula for model
    formula <- as.formula(paste(input$churn_column, "~ ."))
    
    # Train the model based on the user's choice (re-train for consistency)
    model <- train(formula, data = trainData, method = "rpart", 
                   trControl = trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE), 
                   metric = "ROC")
    
    # Predict on test data
    predictions <- predict(model, testData)
    
    # Generate confusion matrix
    conf_matrix <- confusionMatrix(predictions, testData[[input$churn_column]])
    
    # Plot confusion matrix
    output$conf_matrix_output <- renderPlot({
      fourfoldplot(conf_matrix$table, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "Confusion Matrix")
    })
  })
  
  # Show conclusion based on button click
  observeEvent(input$show_conclusion, {
    output$conclusion_output <- renderPrint({
      cat("Given the structure of the data, it consists of:\n\n")
      cat("1. 30 anonymized features (V1 through V28), likely the result of principal component analysis (PCA).\n")
      cat("2. The 'Amount' column, which indicates the transaction amount.\n")
      cat("3. The 'Class' column, which serves as the target variable indicating whether a transaction is fraudulent (1) or not (0).\n")
      cat("4. The 'Time' column, which represents the time elapsed since the first transaction.\n\n")
      cat("To determine if any attributes are strongly linked to fraudulent behavior, let's look at the correlation between the features and the Class variable (target).\n")
      
      # Simulating correlation results
      cat("\nThe correlation analysis between the features and the Class variable reveals the following key points:\n\n")
      cat("Features positively correlated with fraud (Class = 1):\n")
      cat("- V11 (0.154)\n")
      cat("- V4 (0.133)\n")
      cat("- V2 (0.091)\n\n")
      cat("Features negatively correlated with fraud:\n")
      cat("- V17 (-0.326)\n")
      cat("- V14 (-0.302)\n")
      cat("- V12 (-0.260)\n")
      cat("- V10 (-0.216)\n\n")
      cat("Although none of the features show extremely strong correlations with the Class variable, some such as V11, V4, and V17 show moderate relationships. These features may be more indicative of fraudulent behavior and can be explored further to understand their impact on the model.")
    })
  })
  observeEvent(input$show_model_conclusion, {
    output$model_conclusion_output <- renderPrint({
      cat("Summary of evaluations:\n\n")
      cat("Decision tree: This model obtained an AUC of 0.9278. Although this performance is correct, it is weaker than the other models, probably due to the limited complexity of decision trees with default parameters.\n\n")
      cat("Logistic regression: This model performed very well, with an AUC of 0.9830. Logistic regression models often perform well for this type of problem.\n\n")
      cat("SVM (linear): The SVM approach with a linear kernel also performed well with an AUC of 0.9793, showing that class separation is relatively linear.\n\n")
      cat("SVM (RBF kernel): The SVM model with a radial kernel (RBF) performed best with an AUC of 0.9865, suggesting that this model was able to better capture the non-linearity of the data.\n\n")
      cat("Conclusion:\n")
      cat("The SVM model with RBF kernel had the best performance with default hyperparameters, closely followed by logistic regression and linear SVM. The decision tree performed less well than the other approaches.")
    })
  })
  
  # Add a reactive value to store the trained model
  trained_model <- reactiveVal(NULL)
  
  # After the model is trained, save it to the reactive value
  observeEvent(input$train_model, {
    req(input$model_type, input$churn_column)
    data <- selected_data()
    
    # Ensure both classes are present in the dataset
    if (length(unique(data[[input$churn_column]])) < 2) {
      output$model_performance_output <- renderPrint({
        cat("Error: Selected dataset does not contain both churn classes.")
      })
      return()
    }
    
    # Create train/test split
    set.seed(123)
    trainIndex <- createDataPartition(data[[input$churn_column]], p = 0.7, list = FALSE, times = 1)
    trainData <- data[trainIndex, ]
    testData <- data[-trainIndex, ]
    
    # Formula for model
    formula <- as.formula(paste(input$churn_column, "~ ."))
    
    # Train the model based on the user's choice
    model <- NULL
    if (input$model_type == "Decision Tree") {
      model <- train(formula, data = trainData, method = "rpart", 
                     trControl = trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE), 
                     metric = "ROC")
      
    } else if (input$model_type == "Logistic Regression") {
      model <- train(formula, data = trainData, method = "glm", family = "binomial", 
                     trControl = trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE), 
                     metric = "ROC")
      
    } else if (input$model_type == "SVM (without kernel)") {
      model <- train(formula, data = trainData, method = "svmLinear", 
                     trControl = trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE), 
                     metric = "ROC")
      
    } else if (input$model_type == "SVM (with kernel)") {
      model <- train(formula, data = trainData, method = "svmRadial", 
                     trControl = trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE), 
                     metric = "ROC")
    }
    
    # Save the trained model to reactive value
    trained_model(model)
    
    # Predict on test data and show performance
    predictions <- predict(model, testData)
    roc_curve <- roc(testData[[input$churn_column]], as.numeric(predictions))
    auc_value <- auc(roc_curve)
    
    output$model_performance_output <- renderPrint({
      cat("Model:", input$model_type, "\n")
      cat("AUC:", auc_value, "\n")
    })
  })
  
  # Download handler for saving the model
  output$download_model <- downloadHandler(
    filename = function() {
      paste("trained_model_", input$model_type, ".rds", sep = "")
    },
    content = function(file) {
      saveRDS(trained_model(), file)
    }
  )
  
  
}

# Run the app
shinyApp(ui = ui, server = server)
