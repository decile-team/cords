#Learning setting
config = dict(setting= "supervisedlearning",

dataset = dict(name="cifar10",
               datadir= "../../data",
               feature= "dss",
               type= "pre-defined"),

dataloader = dict(shuffle=True,
                  batch_size=20,
                  pin_memory = True),

model = dict(architecture='ResNet18',
             type = 'pre-defined'),

loss = dict(type='CrossEntropyLoss',
            use_sigmoid=False),

optimizer = dict(type="sgd",
                 momentum=0.9,
                 lr=0.01,
                 weight_decay = 5e-4),

scheduler = dict(type = "cosine_annealing",
                 T_max = 300),

dss_strategy = dict(type = "GradMatchPB",
                    fraction = 0.1,
                    select_every = 20),

train_args = dict(num_epochs=300,
                  device="cuda",
                  print_every = 10,
                  results_dir = r'C:\Users\krish\OneDrive - The University of Texas at Dallas\Documents\PyCharmProjects\cords\results',
                  log_args = ["val_loss", "val_acc", "tst_loss", "tst_acc"],
                  return_args = []
                  )
)