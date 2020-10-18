from torch import optim


class Adagrad(optim.Adagrad):
    @classmethod
    def resolve_args(cls, args, model):
        options = {}
        options['lr'] = args.get("learning_rate", 0.01)
        options['lr_decay'] = args.get("lr_decay", 0)
        options['weight_decay'] = args.get("weight_decay", 0)
        return cls(model.parameters(), **options)

class Adam(optim.Adam):
    @classmethod
    def resolve_args(cls,args,model):
        #learning_rate_lm = args.get("learning_rate_lm", 1e-5)
        #kwargs = [{"params": model.parameters(), "lr": 1e-6}]
        options = {}
        options['lr'] = args.get("learning_rate", 3e-4)
        options['weight_decay'] = args.get("weigh_decay", 1e-5)
        return cls(model.parameters(), **options)
