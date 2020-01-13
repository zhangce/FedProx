import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad


def quantize(v, nbit):
    min_ = np.amin(v)
    max_ = np.amax(v)

    nv = ((v - min_) / (max_ - min_) * (2**nbit)).astype(np.int)

    nv = nv.astype(np.float64) / (2**nbit)
    
    nv = nv * (max_ - min_) + min_

    return nv

NBIT = 2

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))


        ### Give all clients the same model
        for c in self.clients:
            c.set_params(self.latest_model)

        client_models = {}
        for c in self.clients:
            client_models[c] = np.copy(c.get_params())


        model_snapshots = {}

        ups = 0
        downs = 0

        last_snapshot = 0
        for i in range(self.num_rounds):

            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                #tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                #tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                #tqdm.write('At round {} training loss: {}     total: {} ups: {} downs: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])), ups + downs, ups, downs)

                test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
                train_loss = np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])
                print (i, train_loss, test_acc, (ups + downs) * NBIT, ups * NBIT, downs * NBIT)

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)


            local_model_last_iters = {}

            csolns = []  # buffer for receiving client solutions
            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                

                ## 1. Communicate the difference between the latest model and the client's local model
                local_client_model = np.copy(c.get_params())
                for i, v in enumerate(local_client_model):
                    d = quantize(self.latest_model[i] - local_client_model[i], NBIT)  
                    local_client_model[i] = local_client_model[i] + d

                    downs = downs + np.size(d)

                local_model_last_iters[idx] = np.copy(local_client_model)
                c.set_params(np.copy(local_client_model))

                # communicate the latest model
                # c.set_params(self.latest_model)

                # solve minimization locally
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            #self.latest_model = self.aggregate(csolns)

            ## Fedavg Aggregation
            #
            total_weight = 0.0
            base = [0]*len(csolns[0][1])
            idx = 0
            for (w, soln) in csolns:  # w is the number of local samples
                total_weight += w
                for i, v in enumerate(soln):

                    local_update = v.astype(np.float64) - local_model_last_iters[idx][i]

                    local_update = quantize(local_update, NBIT)

                    ups = ups + np.size(local_update)

                    base[i] += w * local_update

                idx = idx + 1

            for (i, v) in enumerate(self.latest_model):
                self.latest_model[i] = self.latest_model[i] + base[i] / total_weight

            #self.latest_model = [v / total_weight for v in base]


        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))





