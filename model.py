import numpy as np
import copy

debug = False


def generate_uniform_simplex(no_entries):
    ''' Generates a vector uniformly random in the (no_entries - 1) simplex.

    input: no_entries - number of entries in the vector
    -----
    output: w - numpy array in the (no_entries - 1) simplex.
    '''

    # Get an increasingly sorted list of (no_entries - 1) elements --> dividers of [0, 1]
    dividers = np.random.random(no_entries - 1)
    dividers = np.append(dividers, [0, 1])
    dividers.sort()

    # get the vector in the simplex (weights) given the dividers
    w = np.zeros(no_entries)
    for i in range(no_entries):
        w[i] = dividers[i + 1] - dividers[i]

    return w


class Attributes():
    def __init__(self, config, for_user=False):
        self.config = config

        # True iff these are user atributes (--> all competing values are 1)
        self.for_user = for_user

        self.values = self.generate_values()

    def generate_values(self):
        '''Generates matching and competing attributes for items, and matching attributes for users.
        ---
        ToDo: also make versions for idiosyncratic taste
        '''

        num_attributes = self.config['num_attributes']
        kind_attributes_exp = self.config['kind_attributes_expanded']
        bound = self.config['matching_bound']

        total_num_attributes = sum(num_attributes)
        mean = np.zeros(total_num_attributes)
        cov = self.config['covariance']

        # generate the unrounded values
        attr = np.random.multivariate_normal(mean, cov)

        # conversion from normal attributes to 0/1 for competing and -1/+1 for matching
        def conversion(i):
            '''Conversion from normal attributes to 0/1 for competing and -1/+1 for matching.
            i = position of attribute --> type of attribute, and its value'''
            if 'm' in kind_attributes_exp[i]:
                return 1 if attr[i] > bound else -1
            else:
                if self.for_user:
                    return 1
                else:
                    # return 1 if attr[i] > 0 else 0
                    return attr[i]

        # print(attr)
        attr = [conversion(i) for i in range(total_num_attributes)]
        # print(attr)

        return attr

    def get_maching_attributes(self):
        '''Returns a list of the maching attributes of the CC.'''

        maching = []
        kind_attributes_exp = self.config['kind_attributes_expanded']

        for i, k in enumerate(kind_attributes_exp):
            if 'm' in k:
                maching.append(self.values[i])

        return maching


class User:
    def __init__(self, config, id, list_CCs, attributes=None, weights=None):
        self.config = config
        self.id = id

        # the attributes of the user (the positions for competing attributes are always 1)
        self.attributes = attributes
        if attributes is None:
            self.attributes = Attributes(config, True)

        # the importance the suer gives to each attribute (sums to 1)
        self.weights = weights
        if weights is None:
            self.weights = self.generate_weights()

        # the best CC followed so far
        self.best_followed_CC = None

        # best CCs according to the user (most likely a list of only one)
        self.ranking_CCs = self.get_ranking_CCs(list_CCs)

        # the user is protected if they have a maching attribute with value -1
        self.maching_attr = self.attributes.get_maching_attributes()
        self.protected = -1 in self.maching_attr

    def score(self, c):
        '''Evaluates the score the user associates with content creator c.

        input: c - a content creator
        -----
        ouptut: s - a value the user associates with c
        '''

        kind_attributes_exp = self.config['kind_attributes_expanded']

        # competing attributes add the quality of the CC + the imprtance of the attribute
        # matching attributes boost (reduce) the score by the weight if they (don't) match
        score = 0
        for i, k in enumerate(kind_attributes_exp):
            if 'c' in k:
                score += c.attributes.values[i] * self.weights[i]
            else:
                if self.attributes.values[i] == c.attributes.values[i]:
                    score += self.weights[i]
                else:
                    score -= self.weights[i]

        return score

    def decide_follow(self, c):
        '''Evaluates whether the user wants to follow CC c.

        input: c - a content creator
        ------
        output: bool - decision if it follows c'''

        score = self.score

        # it follows c iff they are better then the best followed so far
        if (self.best_followed_CC is None) or (score(self.best_followed_CC) < score(c)):
            self.best_followed_CC = c
            return True

        return False

    def get_ranking_CCs(self, list_CCs):
        '''Returns a dictionary of the CC and their position in the preference ranking or the user.
        If two are equally on the second position each of them maps to position 2.
        return: dict = {c.id: position_of_c_in_preference_of_u, ...}'''

        sorted_CCs = sorted(
            list_CCs, key=lambda c: self.score(c), reverse=True)
        scores = [self.score(c) for c in list_CCs]

        ranking = {}
        position = -1
        for p, c in enumerate(sorted_CCs):
            # we only increase the position when encountering a new score
            if p == 0 or scores[p - 1] != scores[p]:
                position += 1

            ranking[c.id] = position

        return ranking

    def generate_weights(self):
        '''Generates a weight vector --> the weights of a user for each attribute.
        num_attributes = [#entries for attributes of type 0, ...]
        cumulative_weights = [sum weights of attributes of type 0, ...]
        '''

        num_attributes = self.config['num_attributes']
        cumulative_weights_list = self.config['cumulative_weights']
        prob_cumulative_weights = self.config['prob_cumulative_weights']

        # generate a cumulative weight profile based on the given probabilities in config
        num_w = len(cumulative_weights_list)
        index_w = np.random.choice(range(num_w), p=prob_cumulative_weights)
        cumulative_weights = cumulative_weights_list[index_w]

        # if the we don't have constraints on the cumulative weights, then just get random weights
        if cumulative_weights == -1:
            return generate_uniform_simplex(sum(num_attributes))

        no_types = len(num_attributes)
        weights_parts = []

        for t in range(no_types):
            weights_parts.append(generate_uniform_simplex(
                num_attributes[t]) * cumulative_weights[t])

        weights = np.concatenate(weights_parts)

        return weights


class CC:
    def __init__(self, config, id, attributes=None):
        self.config = config
        self.id = id

        # the attributes of the user (the positions for competing attributes are always 1)
        self.attributes = attributes
        if attributes is None:
            self.attributes = Attributes(config)

        # the content creator is protected if they have a maching attribute with value -1
        self.maching_attr = self.attributes.get_maching_attributes()
        self.protected = -1 in self.maching_attr

    def get_competing_score(self):
        '''Computes the sum of all competing attributes of the CC.'''

        kind_attributes_exp = self.config['kind_attributes_expanded']

        quality = 0
        for i, k in enumerate(kind_attributes_exp):
            if 'c' in k:
                quality += self.attributes.values[i]

        return quality

    def weight_followers_RS(self):
        '''The RS could add biases to content crators.'''

        if self.protected:
            return 1 - self.config['level_bias_RS']
        return 1


class Network:
    '''Class capturing a follower network between from users to items.
    In this version of the code we assumme that each item is a content creator/channel.
    '''

    def __init__(self, config, G=None, favorite=None):
        self.config = config

        num_users = config['num_users']
        num_items = config['num_items']

        self.G = G
        if self.G is None:
            self.G = np.zeros((num_users, num_items), dtype=bool)

        # self.favorite = favorite

        self.num_followers = np.count_nonzero(self.G, axis=0)
        self.num_followees = np.count_nonzero(self.G, axis=1)

    def follow(self, u, c, num_timestep, when_users_found_best):
        '''User u follows content creator c; and updates the Network

        input: u - user
               c - CC
               num_timestep - the iteration number of the platform (int)
               when_users_found_best - a list of length the number of users who keeps the timesteps when each of the user found their best CC (or -1 if they didn't yet)
        '''

        if not self.G[u.id][c.id]:
            if u.decide_follow(c):
                self.G[u.id][c.id] = True
                self.num_followers[c.id] += 1
                self.num_followees[u.id] += 1

                # if c is one of the best CCs for u, then u found their best CC this round
                if u.ranking_CCs[c.id] == 0:
                    when_users_found_best[u.id] = num_timestep

    def is_following(self, u, i):
        return self.G[u][i]


class RS:
    '''Class for the Recommender System (i.e., descoverability  procedure).
    '''

    def __init__(self, config, content_creators):
        self.config = config
        self.biased_weights = np.array(
            [c.weight_followers_RS() for c in content_creators])

    def recommend_random(self, content_creators, biased=False):
        ''' input: content_creators - a list of content creators
                   biased - True if the RS discriminates against protected CCs
        -----
        output: a list of recommendations of CC chosen uniformly at ranodm'''

        num_users = self.config['num_users']
        if biased:
            prob_choice = self.biased_weights / sum(self.biased_weights)
            if debug:
                print('Prob choice RS:', prob_choice)
            return np.random.choice(content_creators, num_users, p=prob_choice)
        return np.random.choice(content_creators, num_users)

    def recommend_PA(self, content_creators, num_followers, biased=False):
        ''' input: content_creators - a list of content creators
                   num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a CC chosen based on PA'''

        num_users = self.config['num_users']
        num_CCs = self.config['num_items']
        prob_choice = num_followers + np.ones(num_CCs)
        if biased:
            prob_choice *= self.biased_weights
        prob_choice /= sum(prob_choice)
        if debug:
            print('Prob choice RS:', prob_choice)
        return np.random.choice(content_creators, num_users, p=prob_choice)

    def recommendable_ExtremePA(self, content_creators, num_followers, biased=False):
        ''' input: content_creators - a list of content creators
                   num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: the CCs that have a maximum number of followers'''

        weighted_num_followers = self.biased_weights * \
            num_followers if biased else num_followers
        max_num_followers = max(weighted_num_followers)

        most_popular_CCs = []
        for c in content_creators:
            if weighted_num_followers[c.id] == max_num_followers:
                most_popular_CCs.append(c)

        return most_popular_CCs

    def recommend_ExtremePA(self, content_creators, num_followers, biased=False):
        ''' input: content_creators - a list of content creators
                   num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a CC chosen based on Extreme PA'''

        num_users = self.config['num_users']

        most_popular_CCs = self.recommendable_ExtremePA(
            content_creators, num_followers, biased)

        if biased:
            prob_choice = np.array([self.biased_weights[c.id]
                                    for c in most_popular_CCs])
            prob_choice = prob_choice / sum(prob_choice)
            if debug:
                print('Prob choice RS:', prob_choice)
            return np.random.choice(most_popular_CCs, num_users, p=prob_choice)
        return np.random.choice(most_popular_CCs, num_users)

    def recommend_AntiPA(self, content_creators, num_followers):
        ''' input: content_creators - a list of content creators
                   num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a CC chosen based on Anti-PA (nodes proportional to exp(-deg) )'''

        num_users = self.config['num_users']
        prob_choice = np.exp(-num_followers) / sum(np.exp(-num_followers))

        return np.random.choice(content_creators, num_users, p=prob_choice)

    def recommend(self, content_creators, num_followers):
        '''A rapper that choses the appropriate RS.

        input: content_creators - a list of content creators
               num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a list of reccommendations (one per user)'''

        if self.config['rs_model'] == 'UR':
            return self.recommend_random(content_creators)
        elif self.config['rs_model'] == 'PA':
            return self.recommend_PA(content_creators, num_followers)
        elif self.config['rs_model'] == 'ExtremePA':
            return self.recommend_ExtremePA(content_creators, num_followers)
        elif self.config['rs_model'] == 'biased_UR':
            return self.recommend_random(content_creators, biased=True)
        elif self.config['rs_model'] == 'biased_PA':
            return self.recommend_PA(content_creators, num_followers, biased=True)
        elif self.config['rs_model'] == 'biased_ExtremePA':
            return self.recommend_ExtremePA(content_creators, num_followers, biased=True)
        elif self.config['rs_model'] == 'AntiPA':
            return self.recommend_AntiPA(content_creators, num_followers)
        elif self.config['rs_model'] == 'PA-AntiPA':
            if np.random.random() < 0.5:
                return self.recommend_PA(content_creators, num_followers)
            return self.recommend_AntiPA(content_creators, num_followers)


class Platform:
    def __init__(self, config):
        self.config = config

        # the platform keeps track of the number of timesteps it has been iterated
        self.timestep = 0

        # make an expanded version of the kind of attributes
        self.config['kind_attributes_expanded'] = []
        for i, k in enumerate(config['kind_attributes']):
            self.config['kind_attributes_expanded'] += [
                k] * config['num_attributes'][i]

        if config['type_attributes'] == 'multidimensional':
            self.config['covariance'] = self.construct_covariance()

        self.network = Network(config)
        self.CCs = []
        self.generate_CCs()
        self.RS = RS(config, self.CCs)
        self.users = [User(config, i, self.CCs)
                      for i in range(config['num_users'])]

        # keep track of the timesteps when users found their best CC
        self.users_found_best = [-1 for u in self.users]
        # keep track of the position of the recommended CC in the ranking of the user
        self.users_rec_pos = []
        # keep track of whether or not the recommende CC had the same maching attributes
        self.rec_same_maching = []
        # keep track of the number of users recommended each CC in each round
        self.num_users_rec_CC = []

        # the users who did not converged yet
        self.id_searching_users = list(range(self.config['num_users']))

        if debug:
            print('The users on the platform have attributes and preferences:')
            for u in self.users:
                print('   ', u.id, u.attributes.values, u.weights)

            print('The CCs on the platform are:')
            for c in self.CCs:
                print('   ', c.id, c.attributes.values)

    def generate_CCs(self):
        '''Generates CCs that have attributes according to the config file'''

        # 1. find if there is a restriction on the % of users of type A (not --> random)
        per_groupA = self.config['%_groupA']
        if per_groupA == -1:
            self.CCs = [CC(self.config, i)
                        for i in range(self.config['num_items'])]
        # 2. else we keep adding CCs of the correct type
        else:
            # define the remaining number of users of each type
            num_typeA = int(self.config['num_items'] * self.config['%_groupA'])
            num_typeB = self.config['num_items'] - num_typeA
            # find the first matching attribute (needs to exist)
            protected_index = self.config['kind_attributes_expanded'].index(
                'm')

            self.CCs = []
            while num_typeA + num_typeB > 0:
                c = CC(self.config,
                       self.config['num_items'] - num_typeA - num_typeB)
                if c.attributes.values[protected_index] == -1 and num_typeA:
                    num_typeA -= 1
                    self.CCs.append(c)
                elif c.attributes.values[protected_index] == 1 and num_typeB:
                    num_typeB -= 1
                    self.CCs.append(c)

    def construct_covariance(self):
        '''Constructs the covariance matrix'''

        num_attributes = self.config['num_attributes']
        dict_cov = self.config['dict_cov']

        no_types = len(num_attributes)
        total_num_attributes = sum(num_attributes)
        cov = np.ones((total_num_attributes, total_num_attributes))

        def type(i):
            '''Given a index, i, returns the type of the attribute on position i.
            (can be done faster)'''
            sum = 0
            for t in range(no_types):
                sum += num_attributes[t]
                if i < sum:
                    return t
            return no_types

        # create the covariance matrix
        for i in range(total_num_attributes):
            # keep the diagonal (variance) 1; so start from i+1
            for j in range(i + 1, total_num_attributes):
                cov[i, j] = dict_cov[(type(i), type(j))]
                cov[j, i] = dict_cov[(type(i), type(j))]

        return cov

    def iterate(self):
        '''Makes one iteration of the platform.
        Used only to update the state of the platform'''

        # 0) the platform starts the next iteration
        self.timestep += 1

        # 1) each user gets a recommendation
        recs = self.RS.recommend(self.CCs, self.network.num_followers)
        # record the position of the recommended CC
        self.users_rec_pos.append(
            [self.users[i].ranking_CCs[c.id] for i, c in enumerate(recs)])
        # record whether the user and the recommended CC mached on type
        self.rec_same_maching.append(
            [int(self.users[i].maching_attr == c.maching_attr) for i, c in enumerate(recs)])
        # record the number of users recommended each CC
        self.num_users_rec_CC.append([0 for c in self.CCs])
        for c in recs:
            self.num_users_rec_CC[-1][c.id] += 1

        # 2) each user decides whether or not to follow the recommended CC
        for u in self.users:
            self.network.follow(
                u, recs[u.id], self.timestep, self.users_found_best)

        if debug:
            print('Recommendations: ', [r.id for r in recs])
            print('New network:', self.network.G)
            print('Number of followers:', self.network.num_followers)
            print('Number of followees:', self.network.num_followees)

    def get_borda_scores(self, rule='original'):
        '''This reflects the global preferences of the consumers (users) on the creators (items).
        1) each consumer ranks all creators (known & unknown);
        2) they give points to each depending on the rule:
             - original --> 1st position gets num_creators points, 2nd gets num_creators-1, ..., 1
        ---
        Returns: array with the borda score for each item
           borda[i] = the score of item i (larger = better)
        '''

        num_items = self.config['num_items']
        num_users = self.config['num_users']

        # get the scores (each raw for a user)
        scores = np.array([[u.score(c) for c in self.CCs] for u in self.users])

        # get the preferences by sorting the scores
        prefs = scores.argsort()  # each row has the respective user's items in increasing order

        borda = np.zeros(num_items)
        if rule == 'original':
            for u in range(num_users):
                for s in range(num_items):
                    borda[prefs[u][s]] += (s + 1)
        elif rule == 'power':
            for u in range(num_users):
                for s in range(num_items):
                    borda[prefs[u][s]] += 1 / (num_items - s)

        return borda

    def get_competing_scores(self):
        ''' Computes the quality of each CC.
        -----
        output: array with the competing socre (i.e., quality with equal weights for dim)
        '''

        quality = []
        for c in self.CCs:
            quality.append(c.get_competing_score())

        return quality

    def update_searching_users(self):
        '''Updates the list of users who are still searching for the best CC.
        i.e. those who did not find the best CC out of the ones that could be recommended
        '''

        if self.config['rs_model'] in ['ExtremePA', 'biased_ExtremePA']:
            # under non-exploratory RSs, the searching users are only the ones who can still find somebody better

            # 1) get the CCs with a maximum number of followers
            most_popular_CCs = self.RS.recommendable_ExtremePA(
                self.CCs, self.network.num_followers, biased='biased' in self.config['rs_model'])

            # 2) find if the user with id i converged
            def converged(i):
                for c in most_popular_CCs:
                    u = self.users[i]
                    if u.score(c) > u.score(u.best_followed_CC):
                        return True
                return False

            # 3) filter users based on whether they can still find a better CC
            self.id_searching_users = list(
                filter(converged, self.id_searching_users))
        else:
            # under exploratory RSs, the searching users are only the ones who did not find the best
            self.id_searching_users = list(
                filter(lambda i: self.users[i].ranking_CCs[self.users[i].best_followed_CC.id] != 0, self.id_searching_users))

    def check_convergence(self):
        # the platform converged if there are no more searching users (users who can find better CCs)
        return len(self.id_searching_users) == 0
