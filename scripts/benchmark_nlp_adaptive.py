import sys

sys.path.append(".")
sys.path.append("..")
from train import TrainClassifier

if __name__ == "__main__":
    config_file = "configs/config_nlp.py"
    # corona
    classifier_corona = TrainClassifier(config_file)
    classifier_corona.configdata["dataset"]["name"] = "corona"
    classifier_corona.configdata["dss_strategy"]["type"] = "GradMatch"
    classifier_corona.configdata["dss_strategy"]["fraction"] = 0.4
    classifier_corona.configdata["dss_strategy"]["select_every"] = 10
    classifier_corona.configdata["dss_strategy"]["lam"] = 0.5
    classifier_corona.configdata["model"]["input_dim"] = 90989
    classifier_corona.configdata["model"]["embed_dim"] = 200
    classifier_corona.configdata["model"]["numclasses"] = 5
    classifier_corona.configdata["model"]["valid"] = False
    classifier_corona.train()

    # # Full
    # config_file = "configs/config_nlp.py"
    #
    # # corona
    # classifier_corona = TrainClassifier(config_file)
    # classifier_corona.configdata["dataset"]["name"] = "corona"
    # classifier_corona.configdata["dss_strategy"]["type"] = "Full"
    # classifier_corona.configdata["dss_strategy"]["fraction"] = 0.1
    # classifier_corona.configdata["dss_strategy"]["select_every"] = 5
    # classifier_corona.configdata["dss_strategy"]["lam"] = 0.5
    # classifier_corona.configdata["model"]["input_dim"] = 90989
    # classifier_corona.configdata["model"]["embed_dim"] = 200
    # classifier_corona.configdata["model"]["numclasses"] = 5
    # classifier_corona.configdata["model"]["valid"] = False
    # classifier_corona.train()
    #
    # # twitter
    # classifier_twitter = TrainClassifier(config_file)
    # classifier_twitter.configdata["dataset"]["name"] = "twitter"
    # classifier_twitter.configdata["dss_strategy"]["type"] = "Full"
    # classifier_twitter.configdata["dss_strategy"]["fraction"] = 0.1
    # classifier_twitter.configdata["dss_strategy"]["select_every"] = 5
    # classifier_twitter.configdata["dss_strategy"]["lam"] = 0.5
    # classifier_twitter.configdata["model"]["input_dim"] = 41302
    # classifier_twitter.configdata["model"]["hidden_units"] = 128
    # classifier_twitter.configdata["model"]["embed_dim"] = 200
    # classifier_twitter.configdata["model"]["numclasses"] = 4
    # classifier_twitter.configdata["model"]["valid"] = False
    # classifier_twitter.train()
    #
    # # ag_news
    # classifier_ag_news = TrainClassifier(config_file)
    # classifier_ag_news.configdata["dataset"]["name"] = "ag_news"
    # classifier_ag_news.configdata["dss_strategy"]["type"] = "Full"
    # classifier_ag_news.configdata["dss_strategy"]["fraction"] = 0.1
    # classifier_ag_news.configdata["dss_strategy"]["select_every"] = 5
    # classifier_ag_news.configdata["dss_strategy"]["lam"] = 0.5
    # classifier_ag_news.configdata["model"]["input_dim"] = 164419
    # classifier_ag_news.configdata["model"]["hidden_units"] = 128
    # classifier_ag_news.configdata["model"]["embed_dim"] = 200
    # classifier_ag_news.configdata["model"]["numclasses"] = 4
    # classifier_ag_news.configdata["model"]["valid"] = False
    # classifier_ag_news.train()
    #
    # # Random-Online
    # for select_every in [3, 5, 20]:
    #     for fraction in [0.1, 0.3, 0.5]:
    #         # corona
    #         classifier_corona = TrainClassifier(config_file)
    #         classifier_corona.configdata["dataset"]["name"] = "corona"
    #         classifier_corona.configdata["dss_strategy"]["type"] = "Random-Online"
    #         classifier_corona.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_corona.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_corona.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_corona.configdata["model"]["input_dim"] = 90989
    #         classifier_corona.configdata["model"]["embed_dim"] = 200
    #         classifier_corona.configdata["model"]["numclasses"] = 5
    #         classifier_corona.configdata["model"]["valid"] = False
    #         classifier_corona.train()
    #
    #         # twitter
    #         classifier_twitter = TrainClassifier(config_file)
    #         classifier_twitter.configdata["dataset"]["name"] = "twitter"
    #         classifier_twitter.configdata["dss_strategy"]["type"] = "Random-Online"
    #         classifier_twitter.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_twitter.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_twitter.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_twitter.configdata["model"]["input_dim"] = 41302
    #         classifier_twitter.configdata["model"]["hidden_units"] = 128
    #         classifier_twitter.configdata["model"]["embed_dim"] = 200
    #         classifier_twitter.configdata["model"]["numclasses"] = 4
    #         classifier_twitter.configdata["model"]["valid"] = False
    #         classifier_twitter.train()
    #
    #         # ag_news
    #         classifier_ag_news = TrainClassifier(config_file)
    #         classifier_ag_news.configdata["dataset"]["name"] = "ag_news"
    #         classifier_ag_news.configdata["dss_strategy"]["type"] = "Random-Online"
    #         classifier_ag_news.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_ag_news.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_ag_news.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_ag_news.configdata["model"]["input_dim"] = 164419
    #         classifier_ag_news.configdata["model"]["hidden_units"] = 128
    #         classifier_ag_news.configdata["model"]["embed_dim"] = 200
    #         classifier_ag_news.configdata["model"]["numclasses"] = 4
    #         classifier_ag_news.configdata["model"]["valid"] = False
    #         classifier_ag_news.train()
    #
    # # GradMatch
    # for select_every in [3, 5, 20]:
    #     for fraction in [0.1, 0.3, 0.5]:
    #         # corona
    #         classifier_corona = TrainClassifier(config_file)
    #         classifier_corona.configdata["dataset"]["name"] = "corona"
    #         classifier_corona.configdata["dss_strategy"]["type"] = "GradMatch"
    #         classifier_corona.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_corona.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_corona.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_corona.configdata["model"]["input_dim"] = 90989
    #         classifier_corona.configdata["model"]["embed_dim"] = 200
    #         classifier_corona.configdata["model"]["numclasses"] = 5
    #         classifier_corona.configdata["model"]["valid"] = False
    #         classifier_corona.train()
    #
    #         # twitter
    #         classifier_twitter = TrainClassifier(config_file)
    #         classifier_twitter.configdata["dataset"]["name"] = "twitter"
    #         classifier_twitter.configdata["dss_strategy"]["type"] = "GradMatch"
    #         classifier_twitter.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_twitter.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_twitter.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_twitter.configdata["model"]["input_dim"] = 41302
    #         classifier_twitter.configdata["model"]["hidden_units"] = 128
    #         classifier_twitter.configdata["model"]["embed_dim"] = 200
    #         classifier_twitter.configdata["model"]["numclasses"] = 4
    #         classifier_twitter.configdata["model"]["valid"] = False
    #         classifier_twitter.train()
    #
    #         # ag_news
    #         classifier_ag_news = TrainClassifier(config_file)
    #         classifier_ag_news.configdata["dataset"]["name"] = "ag_news"
    #         classifier_ag_news.configdata["dss_strategy"]["type"] = "GradMatch"
    #         classifier_ag_news.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_ag_news.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_ag_news.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_ag_news.configdata["model"]["input_dim"] = 164419
    #         classifier_ag_news.configdata["model"]["hidden_units"] = 128
    #         classifier_ag_news.configdata["model"]["embed_dim"] = 200
    #         classifier_ag_news.configdata["model"]["numclasses"] = 4
    #         classifier_ag_news.configdata["model"]["valid"] = False
    #         classifier_ag_news.train()
    #
    # # GradMatch-Warm
    # for select_every in [3, 5, 20]:
    #     for fraction in [0.1, 0.3, 0.5]:
    #         # corona
    #         classifier_corona = TrainClassifier(config_file)
    #         classifier_corona.configdata["dataset"]["name"] = "corona"
    #         classifier_corona.configdata["dss_strategy"]["type"] = "GradMatch-Warm"
    #         classifier_corona.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_corona.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_corona.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_corona.configdata["dss_strategy"]["kappa"] = 0.6
    #         classifier_corona.configdata["model"]["input_dim"] = 90989
    #         classifier_corona.configdata["model"]["embed_dim"] = 200
    #         classifier_corona.configdata["model"]["numclasses"] = 5
    #         classifier_corona.configdata["model"]["valid"] = False
    #         classifier_corona.train()
    #
    #         # twitter
    #         classifier_twitter = TrainClassifier(config_file)
    #         classifier_twitter.configdata["dataset"]["name"] = "twitter"
    #         classifier_twitter.configdata["dss_strategy"]["type"] = "GradMatch-Warm"
    #         classifier_twitter.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_twitter.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_twitter.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_twitter.configdata["dss_strategy"]["kappa"] = 0.6
    #         classifier_twitter.configdata["model"]["input_dim"] = 41302
    #         classifier_twitter.configdata["model"]["hidden_units"] = 128
    #         classifier_twitter.configdata["model"]["embed_dim"] = 200
    #         classifier_twitter.configdata["model"]["numclasses"] = 4
    #         classifier_twitter.configdata["model"]["valid"] = False
    #         classifier_twitter.train()
    #
    #         # ag_news
    #         classifier_ag_news = TrainClassifier(config_file)
    #         classifier_ag_news.configdata["dataset"]["name"] = "ag_news"
    #         classifier_ag_news.configdata["dss_strategy"]["type"] = "GradMatch-Warm"
    #         classifier_ag_news.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_ag_news.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_ag_news.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_ag_news.configdata["dss_strategy"]["kappa"] = 0.6
    #         classifier_ag_news.configdata["model"]["input_dim"] = 164419
    #         classifier_ag_news.configdata["model"]["hidden_units"] = 128
    #         classifier_ag_news.configdata["model"]["embed_dim"] = 200
    #         classifier_ag_news.configdata["model"]["numclasses"] = 4
    #         classifier_ag_news.configdata["model"]["valid"] = False
    #         classifier_ag_news.train()
    #
    # # GradMatchPB
    # for select_every in [3, 5, 20]:
    #     for fraction in [0.1, 0.3, 0.5]:
    #         # corona
    #         classifier_corona = TrainClassifier(config_file)
    #         classifier_corona.configdata["dataset"]["name"] = "corona"
    #         classifier_corona.configdata["dss_strategy"]["type"] = "GradMatchPB"
    #         classifier_corona.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_corona.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_corona.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_corona.configdata["model"]["input_dim"] = 90989
    #         classifier_corona.configdata["model"]["embed_dim"] = 200
    #         classifier_corona.configdata["model"]["numclasses"] = 5
    #         classifier_corona.configdata["model"]["valid"] = False
    #         classifier_corona.train()
    #
    #         # twitter
    #         classifier_twitter = TrainClassifier(config_file)
    #         classifier_twitter.configdata["dataset"]["name"] = "twitter"
    #         classifier_twitter.configdata["dss_strategy"]["type"] = "GradMatchPB"
    #         classifier_twitter.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_twitter.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_twitter.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_twitter.configdata["model"]["input_dim"] = 41302
    #         classifier_twitter.configdata["model"]["hidden_units"] = 128
    #         classifier_twitter.configdata["model"]["embed_dim"] = 200
    #         classifier_twitter.configdata["model"]["numclasses"] = 4
    #         classifier_twitter.configdata["model"]["valid"] = False
    #         classifier_twitter.train()
    #
    #         # ag_news
    #         classifier_ag_news = TrainClassifier(config_file)
    #         classifier_ag_news.configdata["dataset"]["name"] = "ag_news"
    #         classifier_ag_news.configdata["dss_strategy"]["type"] = "GradMatchPB"
    #         classifier_ag_news.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_ag_news.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_ag_news.configdata["dss_strategy"]["lam"] = 0.5
    #         classifier_ag_news.configdata["model"]["input_dim"] = 164419
    #         classifier_ag_news.configdata["model"]["hidden_units"] = 128
    #         classifier_ag_news.configdata["model"]["embed_dim"] = 200
    #         classifier_ag_news.configdata["model"]["numclasses"] = 4
    #         classifier_ag_news.configdata["model"]["valid"] = False
    #         classifier_ag_news.train()
    #
    # # GLISTERPB
    # for select_every in [3, 5, 20]:
    #     for fraction in [0.1, 0.3, 0.5]:
    #         # corona
    #         classifier_corona = TrainClassifier(config_file)
    #         classifier_corona.configdata["dataset"]["name"] = "corona"
    #         classifier_corona.configdata["dss_strategy"]["type"] = "GLISTERPB"
    #         classifier_corona.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_corona.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_corona.configdata["model"]["input_dim"] = 90989
    #         classifier_corona.configdata["model"]["embed_dim"] = 200
    #         classifier_corona.configdata["model"]["numclasses"] = 5
    #         classifier_corona.train()
    #
    #         # twitter
    #         classifier_twitter = TrainClassifier(config_file)
    #         classifier_twitter.configdata["dataset"]["name"] = "twitter"
    #         classifier_twitter.configdata["dss_strategy"]["type"] = "GLISTERPB"
    #         classifier_twitter.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_twitter.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_twitter.configdata["model"]["input_dim"] = 41302
    #         classifier_twitter.configdata["model"]["hidden_units"] = 128
    #         classifier_twitter.configdata["model"]["embed_dim"] = 200
    #         classifier_twitter.configdata["model"]["numclasses"] = 4
    #         classifier_twitter.train()
    #
    #         # ag_news
    #         classifier_ag_news = TrainClassifier(config_file)
    #         classifier_ag_news.configdata["dataset"]["name"] = "ag_news"
    #         classifier_ag_news.configdata["dss_strategy"]["type"] = "GLISTERPB"
    #         classifier_ag_news.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_ag_news.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_ag_news.configdata["model"]["input_dim"] = 164419
    #         classifier_ag_news.configdata["model"]["hidden_units"] = 128
    #         classifier_ag_news.configdata["model"]["embed_dim"] = 200
    #         classifier_ag_news.configdata["model"]["numclasses"] = 4
    #         classifier_ag_news.train()
    #
    # # CRAIG
    # for select_every in [3, 5, 20]:
    #     for fraction in [0.1, 0.3, 0.5]:
    #         # corona
    #         classifier_corona = TrainClassifier(config_file)
    #         classifier_corona.configdata["dataset"]["name"] = "corona"
    #         classifier_corona.configdata["dss_strategy"]["type"] = "CRAIG"
    #         classifier_corona.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_corona.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_corona.configdata["model"]["input_dim"] = 90989
    #         classifier_corona.configdata["model"]["embed_dim"] = 200
    #         classifier_corona.configdata["model"]["numclasses"] = 5
    #         classifier_corona.train()
    #
    #         # twitter
    #         classifier_twitter = TrainClassifier(config_file)
    #         classifier_twitter.configdata["dataset"]["name"] = "twitter"
    #         classifier_twitter.configdata["dss_strategy"]["type"] = "CRAIG"
    #         classifier_twitter.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_twitter.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_twitter.configdata["model"]["input_dim"] = 41302
    #         classifier_twitter.configdata["model"]["hidden_units"] = 128
    #         classifier_twitter.configdata["model"]["embed_dim"] = 200
    #         classifier_twitter.configdata["model"]["numclasses"] = 4
    #         classifier_twitter.train()
    #
    #         # ag_news
    #         classifier_ag_news = TrainClassifier(config_file)
    #         classifier_ag_news.configdata["dataset"]["name"] = "ag_news"
    #         classifier_ag_news.configdata["dss_strategy"]["type"] = "CRAIG"
    #         classifier_ag_news.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_ag_news.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_ag_news.configdata["model"]["input_dim"] = 164419
    #         classifier_ag_news.configdata["model"]["hidden_units"] = 128
    #         classifier_ag_news.configdata["model"]["embed_dim"] = 200
    #         classifier_ag_news.configdata["model"]["numclasses"] = 4
    #         classifier_ag_news.train()
    #
    # # CRAIGPB
    # for select_every in [3, 5, 20]:
    #     for fraction in [0.1, 0.3, 0.5]:
    #         # corona
    #         classifier_corona = TrainClassifier(config_file)
    #         classifier_corona.configdata["dataset"]["name"] = "corona"
    #         classifier_corona.configdata["dss_strategy"]["type"] = "CRAIGPB"
    #         classifier_corona.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_corona.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_corona.configdata["model"]["input_dim"] = 90989
    #         classifier_corona.configdata["model"]["embed_dim"] = 200
    #         classifier_corona.configdata["model"]["numclasses"] = 5
    #         classifier_corona.train()
    #
    #         # twitter
    #         classifier_twitter = TrainClassifier(config_file)
    #         classifier_twitter.configdata["dataset"]["name"] = "twitter"
    #         classifier_twitter.configdata["dss_strategy"]["type"] = "CRAIGPB"
    #         classifier_twitter.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_twitter.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_twitter.configdata["model"]["input_dim"] = 41302
    #         classifier_twitter.configdata["model"]["hidden_units"] = 128
    #         classifier_twitter.configdata["model"]["embed_dim"] = 200
    #         classifier_twitter.configdata["model"]["numclasses"] = 4
    #         classifier_twitter.train()
    #
    #         # ag_news
    #         classifier_ag_news = TrainClassifier(config_file)
    #         classifier_ag_news.configdata["dataset"]["name"] = "ag_news"
    #         classifier_ag_news.configdata["dss_strategy"]["type"] = "CRAIGPB"
    #         classifier_ag_news.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_ag_news.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_ag_news.configdata["model"]["input_dim"] = 164419
    #         classifier_ag_news.configdata["model"]["hidden_units"] = 128
    #         classifier_ag_news.configdata["model"]["embed_dim"] = 200
    #         classifier_ag_news.configdata["model"]["numclasses"] = 4
    #         classifier_ag_news.train()
    #
    # # CRAIG-Warm
    # for select_every in [3, 5, 20]:
    #     for fraction in [0.1, 0.3, 0.5]:
    #         # corona
    #         classifier_corona = TrainClassifier(config_file)
    #         classifier_corona.configdata["dataset"]["name"] = "corona"
    #         classifier_corona.configdata["dss_strategy"]["type"] = "CRAIG-Warm"
    #         classifier_corona.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_corona.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_corona.configdata["dss_strategy"]["kappa"] = 0.6
    #         classifier_corona.configdata["model"]["input_dim"] = 90989
    #         classifier_corona.configdata["model"]["embed_dim"] = 200
    #         classifier_corona.configdata["model"]["numclasses"] = 5
    #         classifier_corona.train()
    #
    #         # twitter
    #         classifier_twitter = TrainClassifier(config_file)
    #         classifier_twitter.configdata["dataset"]["name"] = "twitter"
    #         classifier_twitter.configdata["dss_strategy"]["type"] = "CRAIG-Warm"
    #         classifier_twitter.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_twitter.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_twitter.configdata["dss_strategy"]["kappa"] = 0.6
    #         classifier_twitter.configdata["model"]["input_dim"] = 41302
    #         classifier_twitter.configdata["model"]["hidden_units"] = 128
    #         classifier_twitter.configdata["model"]["embed_dim"] = 200
    #         classifier_twitter.configdata["model"]["numclasses"] = 4
    #         classifier_twitter.train()
    #
    #         # ag_news
    #         classifier_ag_news = TrainClassifier(config_file)
    #         classifier_ag_news.configdata["dataset"]["name"] = "ag_news"
    #         classifier_ag_news.configdata["dss_strategy"]["type"] = "CRAIG-Warm"
    #         classifier_ag_news.configdata["dss_strategy"]["fraction"] = fraction
    #         classifier_ag_news.configdata["dss_strategy"]["select_every"] = select_every
    #         classifier_ag_news.configdata["dss_strategy"]["kappa"] = 0.6
    #         classifier_ag_news.configdata["model"]["input_dim"] = 164419
    #         classifier_ag_news.configdata["model"]["hidden_units"] = 128
    #         classifier_ag_news.configdata["model"]["embed_dim"] = 200
    #         classifier_ag_news.configdata["model"]["numclasses"] = 4
    #         classifier_ag_news.train()
