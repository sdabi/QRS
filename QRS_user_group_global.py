import pennylane as qml
from pennylane import numpy as np
from basic_QRS_class import basic_QRS_circ
from basic_QRS_class import basic_QRS

import defines
import visualiser


class QRS_user_group_global_items(basic_QRS):
    def __init__(self, R, num_of_groups):
        basic_QRS.__init__(self, R)
        self.num_of_groups = num_of_groups

        self.user_params_layers = 3
        self.group_params_layers = 1
        self.global_params_layers = 1
        self.history_params_layers = 2
        self.history_removal_trained = False

        self.user_params = {}
        for user in range(defines._NUM_OF_USERS):
            self.user_params[user] = self.randomize_init_params_QRS(self.user_params_layers)

        self.group_params = {}
        for group in range(self.num_of_groups):
            self.group_params[group] = self.randomize_init_params_QRS(self.group_params_layers)

        self.global_params = {}
        for idx in range(1):
            self.global_params[idx] = self.randomize_init_params_QRS(self.global_params_layers)

        self.hist_removal_params = {}
        for user in range(defines._NUM_OF_USERS):
            self.hist_removal_params[user] = self.randomize_init_params_QRS(self.history_params_layers)

        self.cost_ampl_mask_matrix = self.create_cost_amplification_mask_matrix()
        self.comp_cost_ampl_mask_matrix = self.create_complimantry_cost_amplification_mask_matrix()


        self.heat_matrix = self.create_heat_matrix()

        self.best_group_per_user = {}
        for user in range(defines._NUM_OF_USERS):
            self.best_group_per_user[user] = 0



    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    def train(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.9, beta2=0.999, eps=1e-08)
        print("\n------- TRAINING RECOMMENDATION SYS -------")
        for set in range(10):
            self.find_best_group_per_user()
            self.total_cost.append(0)
            for user in range(defines._NUM_OF_USERS):
                learn_by_other_user = False
                print("Training by user: ", user, end='\r')
                params = self.construct_param_list(self.user_params[user], True,
                                                   self.group_params[self.best_group_per_user[user]], False,
                                                   self.global_params[0], False)


                for t_step in range(10):
                    if not learn_by_other_user:
                        expected_probs_vec = self.expected_probs_vecs[user]
                        cost_ampl_mask = self.cost_ampl_mask_matrix[user]
                    else:
                        relative_user = np.random.choice(defines._NUM_OF_USERS, 1, p=self.heat_matrix[user])[0]
                        expected_probs_vec = self.expected_probs_vecs[relative_user]
                        cost_ampl_mask = self.comp_cost_ampl_mask_matrix[user]

                    learn_by_other_user = not learn_by_other_user
                    params = opt_item_item.step(
                        self.total_cost_basic_QRS_user_items, *params, user=user, expected_probs=expected_probs_vec, cost_ampl_mask=cost_ampl_mask)
                self.update_params(user, params, 'user', 'group', 'global')
            print("")

            params = self.construct_param_list(self.user_params[0], False,
                                               self.group_params[self.best_group_per_user[0]], False,
                                               self.global_params[0], False
                                               )
            probs = basic_QRS_circ(params)
            print("DEBUG for user:", 0)
            interacted_items = self.interacted_items_matrix[0]
            bad_interacted_items = self.bad_interacted_items_matrix[0]
            visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([0])],
                                            is_vec=1,
                                            all_positive=1, digits_after_point=2)



            # self.find_best_group_per_user()
            #
            #
            # # prepare the expected prob vec for every group
            # group_expected_probs_matrix = []
            # for group in range(self.num_of_groups):
            #     group_expected_probs_matrix.append(self.create_group_expected_params(group))
            #
            # # training the groups
            # for groups_set in range(2):
            #     for user in range(defines._NUM_OF_USERS):
            #         params = self.construct_param_list(self.user_params[user], False,
            #                                            self.group_params[self.best_group_per_user[user]], True,
            #                                            self.global_params[0], False)
            #
            #         expected_probs_vec = group_expected_probs_matrix[self.best_group_per_user[user]]
            #
            #         cost_ampl_mask = self.cost_ampl_mask_matrix[user]
            #         for t_step in range(2):
            #             params = opt_item_item.step(
            #                 self.total_cost_basic_QRS_user_items, *params, user=user, expected_probs=expected_probs_vec, cost_ampl_mask=cost_ampl_mask)
            #         self.update_params(user, params, 'user', 'group', 'global')
            # print("")


            print(f"total cost: {self.total_cost[-1]:.3f}\n")


        visualiser.plot_cost_arrs([self.total_cost])
        visualiser.plot_cost_arrs(self.error_per_user)



    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    def train_hist_removal(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.9, beta2=0.999, eps=1e-08)

        hist_removal_expected_probs_matrix = self.create_expected_probs_vecs_for_hist_removal(self.get_QRS_reco_matrix())
        print("\n------- TRAINING RECOMMENDATION SYS -------")
        for set in range(10):

            self.total_cost.append(0)
            for user in range(defines._NUM_OF_USERS):
                # print("Training by user: ", user, end='\r')
                params = self.construct_param_list(self.user_params[user], False,
                                                   self.group_params[self.best_group_per_user[user]], False,
                                                   self.global_params[0], False,
                                                   self.hist_removal_params[user], True)


                expected_probs_vec = hist_removal_expected_probs_matrix[user]
                cost_ampl_mask = np.ones((defines._NUM_OF_ITEMS), requires_grad=False)
                for t_step in range(10):
                    params = opt_item_item.step(
                        self.total_cost_basic_QRS_user_items, *params, user=user, expected_probs=expected_probs_vec, cost_ampl_mask=cost_ampl_mask)
                self.update_params(user, params, 'user', 'group', 'global', 'history')
            print("")

            print(f"total cost: {self.total_cost[-1]:.3f}\n")


        visualiser.plot_cost_arrs([self.total_cost])
        visualiser.plot_cost_arrs(self.error_per_user)
        self.history_removal_trained = True




    # input: list contains: params1, req_grad1, params2 , req_grad2 ....
    def construct_param_list(self, *params_and_gard):
        params_list = []
        for params, req_grad in zip(params_and_gard[::2], params_and_gard[1::2]):
            for layer in params:
                t = np.array(layer.copy())
                t.requires_grad = req_grad
                params_list.append(t)
        return params_list


    def update_params(self, user, params_list, *params_type_list):
        params_list_iterator = 0
        for params_type in params_type_list:
            if params_type == 'user':
                self.user_params[user] = np.array(params_list[params_list_iterator:params_list_iterator+self.user_params_layers])
                params_list_iterator += self.user_params_layers
            if params_type == 'group':
                self.group_params[self.best_group_per_user[user]] = np.array(params_list[params_list_iterator:params_list_iterator+self.group_params_layers])
                params_list_iterator += self.group_params_layers
            if params_type == 'global':
                self.global_params[0] = np.array(params_list[params_list_iterator:params_list_iterator+self.global_params_layers])
                params_list_iterator += self.global_params_layers
            if params_type == 'history':
                self.hist_removal_params[user] = np.array(params_list[params_list_iterator:params_list_iterator + self.history_params_layers])
                params_list_iterator += self.history_params_layers




    def get_recommendation(self, user, uninteracted_items, removed_movie, debug=1):


        if not self.history_removal_trained:
            params = self.construct_param_list(self.user_params[user], False,
                                               self.group_params[self.best_group_per_user[user]], False,
                                               self.global_params[0], False)
        else:
            params = self.construct_param_list(self.user_params[user], False,
                                               self.group_params[self.best_group_per_user[user]], False,
                                               self.global_params[0], False,
                                               self.hist_removal_params[user], False)

        probs = basic_QRS_circ(params)

        # DEBUG
        if debug:
            print("recommendation for user:", user)
            interacted_items = self.interacted_items_matrix[user]
            bad_interacted_items = self.bad_interacted_items_matrix[user]
            visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([removed_movie])],
                                            is_vec=1,
                                            all_positive=1, digits_after_point=2)
        return probs




    def get_QRS_reco_matrix(self):
        QRS_reco_matrix = []
        for user in range(defines._NUM_OF_USERS):
            probs = self.get_recommendation(user, 0, 0, 0)
            QRS_reco_matrix.append(probs)
        return QRS_reco_matrix



    def find_best_group_per_user(self):
        for user in range(defines._NUM_OF_USERS):
            min_error = 2
            best_group = 0

            expected_probs = self.expected_probs_vecs[user]
            for group in range(self.num_of_groups):
                params = self.construct_param_list(self.user_params[user], False,
                                                   self.group_params[group], False,
                                                   self.global_params[0], False)

                probs = basic_QRS_circ(params)

                total_error = sum((expected_probs - probs) ** 2)
                if total_error < min_error:
                    min_error = total_error
                    best_group = group

            if (self.best_group_per_user[user] != best_group):
                print("user", user, "group change", self.best_group_per_user[user], "->", best_group)

            self.best_group_per_user[user] = best_group


    def create_cost_amplification_mask_matrix(self):
        cost_ampl_mask_matrix = []
        for user in range(defines._NUM_OF_USERS):
            cost_ampl_mask = np.ones((defines._NUM_OF_ITEMS))*1
            cost_ampl_mask[self.uninteracted_items_matrix[user]] = 0
            cost_ampl_mask[self.bad_interacted_items_matrix[user]] = 5
            cost_ampl_mask_matrix.append(cost_ampl_mask)
        return cost_ampl_mask_matrix


    def create_complimantry_cost_amplification_mask_matrix(self):
        cost_ampl_mask_matrix = []
        for user in range(defines._NUM_OF_USERS):
            cost_ampl_mask = np.ones((defines._NUM_OF_ITEMS))*1
            cost_ampl_mask[self.interacted_items_matrix[user]] = 0
            cost_ampl_mask[self.bad_interacted_items_matrix[user]] = 0
            cost_ampl_mask_matrix.append(cost_ampl_mask)
        return cost_ampl_mask_matrix



    def create_group_expected_params(self, group):
        expeted_group_vec = np.zeros((defines._NUM_OF_ITEMS))
        for user in range(defines._NUM_OF_USERS):
            if (self.best_group_per_user[user] == group):
                expeted_group_vec[self.interacted_items_matrix[user]] += 1
                expeted_group_vec[self.bad_interacted_items_matrix[user]] -= 1
        expeted_group_vec -= min(expeted_group_vec)
        expeted_group_vec /= sum(expeted_group_vec)
        return expeted_group_vec

    def create_heat_matrix(self):
        heat_matrix = []
        for user in range(defines._NUM_OF_USERS):
            heat_vec = np.zeros((defines._NUM_OF_USERS), requires_grad=False)
            for relative_user in range(defines._NUM_OF_USERS):
                if user == relative_user:
                    continue
                heat_vec[relative_user] = len(set(self.interacted_items_matrix[user]) & set(self.interacted_items_matrix[relative_user]))
            heat_vec /= sum(heat_vec)
            heat_matrix.append(heat_vec)
        return heat_matrix



