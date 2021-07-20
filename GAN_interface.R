library(torch)
library(fairmodels)
library(DALEX)
library(data.table)
data("adult")
adult

sensitive = adult$sex
sensitive = as.integer(sensitive)

sensitive_test = adult_test$sex
sensitive_test = as.integer(sensitive_test)

sensitive2 = adult$race
sensitive2 = as.integer(sensitive2)

sensitive_test2 = adult_test$race
sensitive_test2 = as.integer(sensitive_test2)

numerical=adult[,c(2,4,6,12,13,14)]
numerical_test=adult_test[,c(2,4,6,12,13,14)]

categorical=adult[,c(3,5,7,8,9,15)]
categorical_test=adult_test[,c(3,5,7,8,9,15)]

data <- adult[ , c(2,3,4,5,6,7,8,9,12,13,14,15)]
data$workclass <- as.integer(data$workclass)
data$education  <- as.integer(data$education)
data$marital_status <- as.integer(data$marital_status)
data$occupation <- as.integer(data$occupation)
data$relationship <- as.integer(data$relationship)
data$native_country <- as.integer(data$native_country)
data_matrix=matrix(unlist(data),ncol=12)
data_scaled=scale(data_matrix,center=TRUE,scale=TRUE)

target=adult$salary
target= as.integer(target)
set.seed(7)
train_indices <- sample(1:nrow(adult), 24000)
adult_test<-adult[setdiff(1:nrow(adult), train_indices),]
data_scaled_test<-data_scaled[setdiff(1:nrow(adult), train_indices),]
nrow(adult)
nrow(adult_test)
train_x<-data_scaled[train_indices,]
test_x<-data_scaled[setdiff(1:nrow(data_scaled), train_indices), ]
train_y<-target[train_indices]
test_y<-target[setdiff(1:length(target), train_indices)]
sensitive_train<-sensitive[train_indices]
sensitive_test<-sensitive[setdiff(1:length(sensitive), train_indices)]

#DATA_SET             provided whole data set name
#PRIVILIGED           string indicating privileged group
#PROTECTED            subsetted column of protected values
#train_x              numerical x matrix for training
#train_y              numerical y vector for training
#test_x               numerical x matrix for testing
#test_y               numerical y vector for testing
#sensitive_train      numerical vector of sensitive values for training
#sensitive_test       numerical vector of sensitive values for testing
#PARTITION            size of training data set
#
#BATCH_SIZE           integer size of the batch
#NEURONS_CLF          vectored architecture of the neural network for classifier (each value defines number of neurons in corresponding layer)
#NEURONS_ADV          vectored architecture of the neural network for adversary (each value defines number of neurons in corresponding layer)          
#LEARNING_RATE_CLF    learning rate of classifier
#LEARNING_RATE_ADV    learning rate of adversary
#N_EP_PRECLF          number of epochs for pretrain of classifier
#N_EP_PREADV          number of epochs for pretrain of adversary
#N_EP_CLF             number of epochs for GANs classifier
#N_EP_ADV             number of epochs for GANs adversary
#LAMBDA               the loss function parameter
#N_EP_GAN             number of epochs for GAN


#--------------------------- 2) Classifier ---------------------------# 

BATCH_SIZE = 200; NEURONS_CLF = c(32,32,32); NEURONS_ADV = c(32,32,32);
LEARNING_RATE_CLF = 0.01; LEARNING_RATE_ADV =0.001; N_EP_PRECLF = 5; N_EP_PREADV = 10; N_EP_CLF = 1; N_EP_ADV = 1; LAMBDA = 10000; N_EP_GAN = 100; 
PARTITION = 0.7; DATA_SET = adult_test; PRIVILIGED = "Male"; PROTECTED = adult_test$sex;

EPOCH_PRINT=5
?explain
# sets device to cuda if available (however cuda is still not implemented properly)
device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"
# loads data set with dataset and data loader functions, returns 2 data sets and 2 data loaders for train and test respectively
dataset_loader <- function(train_x,train_y,test_x,test_y,batch_size=5){
  new_dataset <- dataset(
    
    name = "new_dataset",
    
    initialize = function(df,y2) {
      
      df <- na.omit(df) 
      x_cont <- df
      self$x_cont <- torch_tensor(x_cont)
      self$y <- torch_tensor(y2,dtype = torch_long())
      
    },
    
    .getitem = function(i) {
      list(x_cont = self$x_cont[i, ], y=self$y[i])
      
    },
    
    .length = function() {
      self$y$size()[[1]]
    }
    
  )
  
  train_ds <- new_dataset(train_x,train_y)
  test_ds <- new_dataset(test_x,test_y)
  train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
  test_dl <- test_ds %>% dataloader(batch_size = batch_size, shuffle = FALSE)
  
  return(list("train_ds" = train_ds,"test_ds"=test_ds,"train_dl"=train_dl,"test_dl"=test_dl))
}

dsl <- dataset_loader(train_x,train_y,test_x,test_y,batch_size = BATCH_SIZE)
# creates the neural network and model WARNING you have to set correct number of neurons in the for loop
create_model <- function(train_x,train_y,neurons,dimensions){ #ustaw liczbe neuronów
  net <- nn_module(
    "net",
    initialize = function(n_cont, Neurons, output_dim) {
      torch_manual_seed(7)
      self$fc1<-nn_linear(n_cont, Neurons[1])
      #self$do1<-nn_dropout(p = 0)
      for (i in 2:length(Neurons)){
        str<-paste("self$fc",i," <- nn_linear(Neurons[",i-1,"]", ",Neurons[",i,"])",sep="")
        eval(parse(text=str))
      }
      # for (i in 2:length(Neurons)){
      #   str<-paste("self$do",i," <- nn_dropout(p = 0.5)",sep="")
      #   eval(parse(text=str))
      # }
      self$output <- nn_linear(Neurons[length(Neurons)], output_dim)
      
    },
    forward = function(x_cont) {
      all <- torch_cat(x_cont,dim=2)
      for (i in 1:2){
        str<-paste("all<-all %>% self$fc",i,"() %>% 
                    nnf_relu()",sep="")
        eval(parse(text=str))
      }
      all %>% self$output() %>% nnf_softmax(dim = dimensions)
      
    }
  )
  model <- net(
    n_cont = ncol(data.frame(train_x)),
    Neurons = neurons,
    output_dim = length(levels(factor(train_y))) 
  )
  
}

clf_model <- create_model(train_x,train_y,neurons = NEURONS_CLF, dimensions = 2)
# evaluate classifier by calculating accuracy
eval_clf <- function(model,test_ds){
  model$eval()
  test_dl <- test_ds %>% dataloader(batch_size = test_ds$.length(), shuffle = FALSE)
  iter <- test_dl$.iter()
  b <- iter$.next()
  output <- model(b$x_cont)
  preds <- output$to(device = "cpu") %>% as.array()
  preds <- ifelse(preds[,1] < preds[,2], 2, 1)
  comp_df <- data.frame(preds = preds, y = b$y %>% as_array())
  num_correct <- sum(comp_df$preds == comp_df$y)
  num_total <- nrow(comp_df)
  accuracy <- num_correct/num_total
  return(accuracy)
}
# compiles the given model, prints the loss metrics and returns train and test losses
compile_model <- function(n_epochs=15,model,train_dl,test_dl,test_ds=NULL,loss_type=1, lambda=0, test_loss=0, train_loss=0, learnig_rate){
  
  optimizer <- optim_adam(model$parameters, lr = learnig_rate)
  
  if (loss_type <= 1){
    calc_loss <- function(lambda,output,batch,adv_loss){
      loss <- nnf_cross_entropy(output, batch)
      return(loss)
    }
  } else if (loss_type == 2){
    calc_loss <- function(lambda,output,batch,adv_loss){
      loss <- nnf_cross_entropy(output, batch)*lambda
      return(loss)
    }
  } else if(loss_type == 3){
    calc_loss <- function(lambda,output,batch,adv_loss){ 
      loss <- nnf_cross_entropy(output, batch)-adv_loss
      return(loss)
    }
  }
  
  train_eval <- function(model,train_dl,test_dl,optimizer,train_loss,test_loss,lambda){ ## finish
    model$train()
    train_losses <- c()  
    coro::loop(for (b in train_dl) {
      optimizer$zero_grad()
      output <- model(b$x_cont)
      loss <- calc_loss(lambda,output,b$y,test_loss)
      loss$backward()
      optimizer$step()
      train_losses <- c(train_losses, loss$item())
    })
    model$eval()
    valid_losses <- c()
    coro::loop(for (b in test_dl) {
      output <- model(b$x_cont)
      loss <- calc_loss(lambda,output,b$y,train_loss)
      valid_losses <- c(valid_losses, loss$item())
    })
    
    return(list("train_loss"=mean(train_losses), "test_loss"= mean(valid_losses)))
  }
  
  if(loss_type == 0){
    for (epoch in 1:n_epochs) {
      losses <- train_eval(model,train_dl,test_dl,optimizer,train_loss,test_loss,lambda)
      acc<-eval_clf(model, test_ds)
      cat(sprintf("Preadversary Loss at epoch %d: training: %3.3f, validation: %3.3f, accuracy: %3.3f\n", epoch, losses$train_loss, losses$test_loss, acc))
    }
  }
  if(loss_type == 1){
    for (epoch in 1:n_epochs) {
      losses <- train_eval(model,train_dl,test_dl,optimizer,train_loss,test_loss,lambda)
      acc<-eval_clf(model, test_ds)
      cat(sprintf("Preclassifier Loss at epoch %d: training: %3.3f, validation: %3.3f, accuracy: %3.3f\n", epoch, losses$train_loss, losses$test_loss, acc))
    }
  }
  if(loss_type == 2){
    for (epoch in 1:n_epochs) {
      losses <- train_eval(model,train_dl,test_dl,optimizer,train_loss,test_loss,lambda)
      acc<-eval_clf(model, test_ds)
      cat(sprintf("Adversary Loss at epoch %d: training: %3.3f, validation: %3.3f, accuracy: %3.3f\n", epoch, losses$train_loss, losses$test_loss, acc))
    }
  }
  if(loss_type == 3){
    iter <- train_dl$.iter()
    b <- iter$.next()
    model$train()
    train_losses <- c()  
    optimizer$zero_grad()
    output <- model(b$x_cont)
    loss <- calc_loss(lambda,output,b$y,test_loss)
    loss$backward()
    optimizer$step()
    train_losses <- c(train_losses, loss$item())
    model$eval()
    
    valid_losses <- c()
    iter <- test_dl$.iter()
    b <- iter$.next()
    output <- model(b$x_cont)
    loss <- calc_loss(lambda,output,b$y,train_loss)
    valid_losses <- c(valid_losses, loss$item())
    
    
    losses <- (list("train_loss"=mean(train_losses), "test_loss"= mean(valid_losses)))
    
    for (epoch in 1:n_epochs) {
      losses <- train_eval(model,train_dl,test_dl,optimizer,train_loss,test_loss,lambda)
      acc<-eval_clf(model, test_ds)
      cat(sprintf("Classifier Loss at epoch %d: training: %3.3f, validation: %3.3f, accuracy: %3.3f\n", epoch, losses$train_loss, losses$test_loss, acc))
    }
  }
  return(list("train_loss"=losses$train_loss, "test_loss"= losses$test_loss))
}

compile_model(n_epochs = N_EP_PRECLF, model = clf_model, train_dl = dsl$train_dl, test_dl = dsl$test_dl,test_ds=dsl$test_ds,loss_type = 1,learnig_rate = LEARNING_RATE_CLF)
# makes 0/1 prediction of classes
make_preds <- function(model,test_ds){
  model$eval()
  test_dl <- test_ds %>% dataloader(batch_size = test_ds$.length(), shuffle = FALSE)
  iter <- test_dl$.iter()
  b <- iter$.next()
  output <- model(b$x_cont)
  #print(output)
  preds <- output$to(device = "cpu") %>% as.array()
  preds <- ifelse(preds[,1] < preds[,2], 2, 1)
  return(preds)
  
}

preds<-make_preds(model = clf_model, test_ds = dsl$test_ds)
# makes probability prediction of classes
make_preds_prob <- function(model,test_ds){
  model$eval()
  test_dl <- test_ds %>% dataloader(batch_size = test_ds$.length(), shuffle = FALSE)
  iter <- test_dl$.iter()
  b <- iter$.next()
  output <- model(b$x_cont)
  preds <- output$to(device = "cpu") %>% as.array()
  return(preds)
  
}

p_preds <- make_preds_prob(clf_model,dsl$train_ds)

eval_clf(model = clf_model, test_ds = dsl$test_ds)
# creates DALEX fairness explainer and calculates the fairness check, returns fobject - fairness object
GAN_explainer <- function(target,clf_model,data_set,protected,privileged){
  
  y_numeric <- as.numeric(target)-1
  custom_predict <- function(mmodel, newdata) {
    pp<-make_preds_prob(model = mmodel, test_ds = dataset_loader(data_scaled_test,test_y,data_scaled_test,test_y)$test_ds)
    pp[,2]
  }
  aps_model_exp <- DALEX::explain(clf_model, data = data_set, y = y_numeric,
                                  predict_function = custom_predict, 
                                  type = 'classification')
  fobject <- fairness_check(aps_model_exp,
                            protected = protected,
                            privileged = privileged)
  print(fobject)
  print(model_performance(aps_model_exp))
  plot(fobject)
  return(fobject)
}

exp1 <-  GAN_explainer(test_y,clf_model,DATA_SET,PROTECTED,PRIVILIGED)

plot(exp1)

#--------------------------- 3) Adversary ---------------------------#

# prepares data to adversarial model, by making proportional samples of two classes and turning all of them to train and test x and y
prepare_to_adv <- function(preds, sensitive, PARTITION){
  df <- data.frame(preds = preds, sensitive = sensitive)
  M <- min(table(sensitive))
  df_new <- df[df$sensitive == 1, ][1:M, ]
  df_new <- rbind(df_new, df[df$sensitive == 2, ][1:M, ])
  preds <- df_new$preds
  sensitive <- df_new$sensitive
  set.seed(123)
  train_indices <- sample(1:length(preds),  length(preds) * PARTITION)
  train_x <- as.numeric(preds[train_indices])
  train_x <- matrix(train_x, ncol=1)
  train_y <- sensitive[train_indices]
  test_x <- as.numeric(preds[setdiff(1:length(preds), train_indices)])
  test_x <- matrix(test_x, ncol=1)
  test_y <- sensitive[setdiff(1:length(sensitive), train_indices)]
  return(list("train_x"=train_x,"train_y"=train_y,"test_x"=test_x,"test_y"=test_y))
}

prepared_data <- prepare_to_adv(p_preds[,2],sensitive_train,PARTITION)

dsl_adv <- dataset_loader(prepared_data$train_x,prepared_data$train_y,prepared_data$test_x,prepared_data$test_y,batch_size = BATCH_SIZE)

adv_model <- create_model(prepared_data$train_x,prepared_data$train_y,neurons = NEURONS_ADV, dimensions = 2)

compile_model(n_epochs = N_EP_PREADV, model = adv_model, train_dl = dsl_adv$train_dl, test_dl = dsl_adv$test_dl, test_ds = dsl_adv$test_ds, loss_type = 0, learnig_rate = LEARNING_RATE_ADV)

prepared_data$test_y

make_preds(adv_model,dsl_adv$test_ds)

eval_clf(adv_model,dsl_adv$test_ds)

#--------------------------- 4) GAN ---------------------------#
# single epoch for GAN training simultaneously the classifier and adversarial
fair_train <- function(clf_model,adv_model, train_dl, test_dl, train_dl2, test_dl2, train_ds,test_ds, test_ds2, lambda, sensitive_train,sensitive_test,train_y,PARTITION){
  # bierzemy srednia funkcje straty dla traina i testa z adwersarza, aby potem przekazac je do klasyfikatora
  losses <- compile_model(N_EP_ADV,adv_model,train_dl2, test_dl2,test_ds = test_ds2,loss_type = 2, lambda = lambda,learnig_rate = LEARNING_RATE_ADV)
  # compilujemy klasyfikator wraz z oboma funkcjami straty (czyli od lossa odejmujemy to co dostaniemy w poprzednim kroku)
  compile_model(N_EP_CLF,clf_model,train_dl,test_dl,test_ds=test_ds,loss_type = 3,lambda = lambda, train_loss = losses$train_loss, test_loss = losses$test_loss,learnig_rate = LEARNING_RATE_CLF)
  # tworzymy predykcje z klasyfikatora
  preds <- make_preds_prob(clf_model,train_ds)
  # tworzmy dane do adwersarza, one beda zbalansowane i na nowo losowane
  prepared_data <- prepare_to_adv(preds[,2],sensitive_train,PARTITION)
  
  dsl_adv <- dataset_loader(prepared_data$train_x,prepared_data$train_y,prepared_data$test_x,prepared_data$test_y)
  
  return(dsl_adv)
}
# GAN training function, with right amount of epochs
GAN_train <- function(epochs, clf_model, adv_model, dsl, dsl_adv, lambda, sensitive_train,sensitive_test,train_y,PARTITION,EPOCH_PRINT){
  for (epoch in 1:epochs){ 
    cat(sprintf("GAN epoch %d \n", epoch))
    dsl_adv <- fair_train(clf_model, adv_model, dsl$train_dl, dsl$test_dl, dsl_adv$train_dl, dsl_adv$test_dl, dsl$train_ds, dsl$test_ds, dsl_adv$test_ds, lambda,sensitive_train,sensitive_test,train_y,PARTITION)
    if(epoch/EPOCH_PRINT==as.integer(epoch/EPOCH_PRINT)){
      exp1 <-  GAN_explainer(test_y,clf_model,DATA_SET,PROTECTED,PRIVILIGED)
      plot(exp1)
    }
  }
}

GAN_train(N_EP_GAN, clf_model, adv_model, dsl, dsl_adv, lambda=LAMBDA, sensitive_train,sensitive_test,train_y,PARTITION,EPOCH_PRINT)

exp <-  GAN_explainer(test_y,clf_model,DATA_SET,PROTECTED,PRIVILIGED)

plot(exp)
