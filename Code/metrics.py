import math
import pandas as pd
#from ml_metrics import mapk
import numpy as np
from itertools import combinations
import sys


class MetronAtK(object):
    def __init__(self, top_k, loo_eval):
        self._top_k = top_k
        self.loo_eval = loo_eval
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        assert isinstance(subjects, list)
        if self.loo_eval == True:
            test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
            neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
            # the golden set
            test = pd.DataFrame({'user': test_users,
                                 'test_item': test_items,
                                 'test_score': test_scores})
            # the full set
            full = pd.DataFrame({'user': neg_users + test_users,
                                 'item': neg_items + test_items,
                                 'score': neg_scores + test_scores})
            full = pd.merge(full, test, on=['user'], how='left')
            # rank the items according to the scores for each user
            full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
            full.sort_values(['user', 'rank'], inplace=True)
            self._subjects = full
        else:
            test_users, test_items, test_true, test_output = subjects[0], subjects[1], subjects[2], subjects[3]
            # the golden set
            full = pd.DataFrame({'user': test_users,
                                 'test_item': test_items,
                                 'test_true': test_true,
                                 'test_output': test_output})

            # rank the items according to the scores for each user
            full['rank'] = full.groupby('user')['test_output'].rank(method='first', ascending=False)
            full['rank_true'] = full.groupby('user')['test_true'].rank(method='first', ascending=False)
            full.sort_values(['user', 'rank'], inplace=True)
            self._subjects = full

    def cal_ndcg(self):
        """NDCG@K for explicit evaluation"""
        full, top_k = self._subjects, self._top_k
        topp_k = full[full['rank_true'] <= top_k].copy()
        topp_k['idcg_unit'] = topp_k['rank_true'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        topp_k['idcg'] = topp_k.groupby(['user'])['idcg_unit'].transform('sum')

        test_in_top_k = topp_k[topp_k['rank'] <= top_k].copy()
        test_in_top_k['dcg_unit'] = test_in_top_k['rank'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        test_in_top_k['dcg'] = test_in_top_k.groupby(['user'])['dcg_unit'].transform('sum')
        test_in_top_k['ndcg'] = test_in_top_k['dcg'] / topp_k['idcg']
        ndcg = np.sum(test_in_top_k.groupby(['user'])['ndcg'].max()) / len(full['user'].unique())
        del (topp_k, test_in_top_k)
        return ndcg

    def cal_map_at_k(self):
        """MAP@K for explicit evaluation"""
        full, top_k = self._subjects, self._top_k
        users = list(dict.fromkeys(list(full['user'])))
        actual = [list(full[(full['user'] == user) & (full['rank_true'] <= top_k)]['test_item']) for user in users]
        predicted = [list(full[(full['user'] == user) & (full['rank'] <= top_k)]['test_item']) for user in users]
        # print(actual)
        # print(predicted)
        print("#############")
        return 10.999
        # i = 0
        # for i in 

        #return mapk(actual, predicted, k=top_k)

    def cal_hit_ratio_loo(self):
        """HR@K for Leave-One-Out evaluation"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        if len(test_in_top_k) == 0:
            return 0
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg_loo(self):
        """NDCG@K for Leave-One-Out evaluation"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()

    def cal_mep(self, explainability_matrix, theta):
        """Mean Explainability Precision at cutoff top_k and threshold theta"""
        full, top_k = self._subjects, self._top_k
        if self.loo_eval == True:
            # VECTORISED (rexbench): was
            #   full[['user','item']].apply(lambda x: explainability_matrix[x[0], x[1]].item(), axis=1)
            # a row-by-row Python apply over every (user, item) pair in the evaluation set.
            # It made evaluate() 1090x slower than training an epoch (measured on ml100k:
            # 436s vs 0.4s) and scales with the number of users, so electronics/rentrunway
            # (100k+ users) were effectively unrunnable. NumPy fancy indexing computes the
            # identical values -- verified element-wise -- 482-608x faster.
            full['exp_score'] = explainability_matrix[full['user'].to_numpy(), full['item'].to_numpy()]
        else:
            full['exp_score'] = explainability_matrix[full['user'].to_numpy(), full['test_item'].to_numpy()]
        full['exp_and_rec'] = ((full['exp_score'] > theta) & (full['rank'] <= top_k)) * 1
        full['topN'] = (full['rank'] <= top_k) * 1
        return np.mean(full.groupby('user')['exp_and_rec'].sum() / full.groupby('user')['topN'].sum())

    def cal_weighted_mep(self, explainability_matrix, theta):
        """Weighted Mean Explainability Precision at cutoff top_k and threshold theta"""
        full, top_k = self._subjects, self._top_k
        if self.loo_eval == True:
            # VECTORISED (rexbench): was
            #   full[['user','item']].apply(lambda x: explainability_matrix[x[0], x[1]].item(), axis=1)
            # a row-by-row Python apply over every (user, item) pair in the evaluation set.
            # It made evaluate() 1090x slower than training an epoch (measured on ml100k:
            # 436s vs 0.4s) and scales with the number of users, so electronics/rentrunway
            # (100k+ users) were effectively unrunnable. NumPy fancy indexing computes the
            # identical values -- verified element-wise -- 482-608x faster.
            full['exp_score'] = explainability_matrix[full['user'].to_numpy(), full['item'].to_numpy()]
        else:
            full['exp_score'] = explainability_matrix[full['user'].to_numpy(), full['test_item'].to_numpy()]
        full['exp_and_rec'] = ((full['exp_score'] > theta) & (full['rank'] <= top_k)) * 1 * (full['exp_score'])
        full['topN'] = (full['rank'] <= top_k) * 1
        return np.mean(full.groupby('user')['exp_and_rec'].sum() / full.groupby('user')['topN'].sum())

    def avg_popularity(self, popularity_vector):
        """Average popularity of top_k recommended items"""
        full, top_k = self._subjects, self._top_k
        if self.loo_eval == True:
            recommended_items = list(full.loc[full['rank'] <= top_k]['item'])
        else:
            recommended_items = list(full.loc[full['rank'] <= top_k]['test_item'])
        return np.mean([popularity_vector[i] for i in recommended_items])

    def efd(self, popularity_vector):
        """Expected Free Discovery (EFD) in top_k recommended items"""
        full, top_k = self._subjects, self._top_k
        if self.loo_eval == True:
            recommended_items = list(full.loc[full['rank'] <= top_k]['item'])
        else:
            recommended_items = list(full.loc[full['rank'] <= top_k]['test_item'])
        return np.mean([- np.log2(popularity_vector[i] + sys.float_info.epsilon) for i in recommended_items])

    def avg_pairwise_similarity(self, item_similarity_matrix):
        """Average Pairwise Similarity of top_k recommended items"""
        full, top_k = self._subjects, self._top_k
        full = full.loc[full['rank'] <= top_k]
        # VECTORISED (rexbench): was
        #   users = list(dict.fromkeys(list(full['user'])))
        #   rec_items_for_users = [list(full.loc[full['user'] == u]['<col>']) for u in users]
        # which re-scanned the WHOLE frame once per user -- O(users x rows), i.e. O(users^2)
        # since rows ~ users * top_k. Measured exponent on real data: users^1.95. At 132,393
        # users (electronics) that is ~1.8e12 operations, projecting to 562 h for a single
        # fit. groupby does it in one pass.
        #
        # Exact-equivalence notes: groupby(sort=False) yields groups in first-appearance
        # order, matching dict.fromkeys, and preserves row order within each group -- so the
        # resulting list of lists is identical. The `len(x) > 1` filter stays in the `else`
        # branch ONLY, exactly as before: under loo_eval=True a user with a single top-k item
        # still yields an empty combinations() and therefore nan, and that behaviour is
        # preserved rather than quietly fixed (top_k ranks guarantee it cannot trigger here).
        _col = 'item' if self.loo_eval == True else 'test_item'
        rec_items_for_users = list(full.groupby('user', sort=False)[_col].apply(list))
        if self.loo_eval != True:
            rec_items_for_users = [x for x in rec_items_for_users if len(x) > 1]
        item_combinations = [set(combinations(rec_items_for_user, 2)) for rec_items_for_user in rec_items_for_users]
        return np.mean([np.mean([item_similarity_matrix[i, j] for (i, j) in item_combinations[k]]) for k in range(len(item_combinations))])