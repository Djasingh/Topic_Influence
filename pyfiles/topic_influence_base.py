#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import random
from scipy.stats import entropy  #Kullback-Leibler divergence
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import find_peaks
from collections import Counter


# In[26]:


get_ipython().run_line_magic('run', 'group_creation.ipynb')


# In[27]:


get_ipython().run_line_magic('run', 'topic_model_training_with_mathscinet.ipynb')


# In[28]:


def researcher_publication(publication):
    publ_count = publication.groupby(["mgp_id"])['yw_publication_count'].apply(sum).reset_index(name='total_publ').copy()
    mgpid2publ_count = dict(zip(publ_count["mgp_id"], publ_count["total_publ"]))
    return mgpid2publ_count


# In[29]:


def publication_extract(group, publication):
    grp_titles = None
    if type(group)==list or type(group)==tuple:
        grp_publ= publication[publication["mgp_id"].isin(group)].copy()
        grp_publ["yearwise_titles"]= grp_publ["yearwise_titles"].apply(lambda x : " ".join(x))
        yearwise_grp_publ = grp_publ.groupby(["mgp_id","publication_year"])['yearwise_titles'].apply(list).reset_index(name='group_yearwise_titles').copy()
        yearwise_grp_publ.sort_values('publication_year', inplace=True)
        grp_titles = [(" ".join(yearwise), year) for yearwise, year in yearwise_grp_publ[["group_yearwise_titles","publication_year"]].values]
    else:
        grp_publ= publication[publication["mgp_id"]==group].copy()
        grp_publ["yearwise_titles"]= grp_publ["yearwise_titles"].apply(lambda x : " ".join(x))
        yearwise_grp_publ = grp_publ.groupby(["mgp_id","publication_year"])['yearwise_titles'].apply(list).reset_index(name='group_yearwise_titles').copy()
        yearwise_grp_publ.sort_values('publication_year', inplace=True)
        grp_titles = [(" ".join(yearwise), year) for yearwise, year in yearwise_grp_publ[["group_yearwise_titles","publication_year"]].values]
    return grp_titles


# In[30]:


def constrained_publication_extract(group, publication, years):
    grp_titles = None
    if type(group)==list or type(group)==tuple:
        grp_publ= publication[publication["mgp_id"].isin(group)].copy()
        tmp = []
        advisor_publ = grp_publ[(grp_publ["mgp_id"]==group[0]) & (grp_publ["publication_year"] >= years[0])]
        if advisor_publ.shape[0] > 0: 
            tmp.append(advisor_publ)
        for grp_memeber, year in zip(group[1:],years[1:]):
            advisees_publ = grp_publ[(grp_publ["mgp_id"]==grp_memeber) & (grp_publ["publication_year"] <= year)]
            if advisees_publ.shape[0] > 0:
                tmp.append(advisees_publ)
        if len(tmp) > 0:
            grp_data = pd.concat(tmp)
            grp_data["yearwise_titles"]= grp_data["yearwise_titles"].apply(lambda x : " ".join(x))
            yearwise_grp_publ = grp_data.groupby(["mgp_id","publication_year"])['yearwise_titles'].apply(list).reset_index(name='group_yearwise_titles').copy()
            yearwise_grp_publ.sort_values('publication_year', inplace=True)
            grp_titles = [(" ".join(yearwise), year) for yearwise, year in yearwise_grp_publ[["group_yearwise_titles","publication_year"]].values]
        else:
            grp_titles = []
    return grp_titles


# In[31]:


# publication = publicaton_data(stem_title=True)
# mgp_nodes, mgp_edges = read_MGP()
# mgpid2_cy = dict(zip(mgp_nodes["Id"],mgp_nodes["Year"]))
# mgp_nodes["mathscinet_id"] = mgp_nodes["MSN"].apply(lambda x : x.rsplit("/")[-1] if type(x)==str else x)
# mathscinet2mgpid = dict(zip(mgp_nodes["mathscinet_id"], mgp_nodes["Id"]))
# publication["mgp_id"]=publication["author_id"].map(mathscinet2mgpid)
# publication["publication_year"] =  pd.to_datetime(publication["publication_year"])
# publication["publication_year"] = publication["publication_year"].dt.year
# filtered_groups, filtered_grp_member_cy, mgpid2publ_count, group_publication_titles = group_info()


# In[32]:


#publication[(publication["mgp_id"]==3) & (publication["publication_year"] >= 1939.0)]


# In[33]:


#constrained_publication_extract((258,)+filtered_groups[258],publication, filtered_grp_member_cy[0])


# In[34]:


def group_info(group_size=5):
    mgp_nodes, mgp_edges = read_MGP()
    mgpid2_cy = dict(zip(mgp_nodes["Id"],mgp_nodes["Year"]))
    mgp_nodes["mathscinet_id"] = mgp_nodes["MSN"].apply(lambda x : x.rsplit("/")[-1] if type(x)==str else x)
    mathscinet2mgpid = dict(zip(mgp_nodes["mathscinet_id"], mgp_nodes["Id"]))
    groups = group_Formation(mgp_nodes, mgp_edges)
    filtered_groups = group_filter(groups, group_size)
    publication = publicaton_data(stem_title=True)
    #publication["yearwise_titles"]= publication["yearwise_titles"].apply(lambda x : " ".join(x))
    publication["mgp_id"]=publication["author_id"].map(mathscinet2mgpid)
    publication["yw_publication_count"] = publication["yearwise_titles"].apply(lambda x:len(x))
    mgpid2publ_count = researcher_publication(publication)
    filtered_grp_member = [(k,)+v for k,v in filtered_groups.items()]
    filtered_grp_member_cy = [[mgpid2_cy.get(member, None) for member in grp] for grp in  filtered_grp_member] #completion year
    publication["publication_year"] =  pd.to_datetime(publication["publication_year"])
    publication["publication_year"] = publication["publication_year"].dt.year
    group_publication_titles = {grp[0]: constrained_publication_extract(grp, publication, grp_cy) for grp, grp_cy in zip(filtered_grp_member, filtered_grp_member_cy)}
    #group_publication_titles = {grp[0]: publication_extract(grp, publication) for grp in filtered_grp_member}
    return filtered_groups, filtered_grp_member_cy, mgpid2publ_count, group_publication_titles, mgp_nodes, mgp_edges


# In[35]:


def topic_distributions_over_interval(group_publication, topic_model, id2word, preprocess, year_interval=2):
    group_leader = group_publication[0]
    publication  = group_publication[1]
    first_pub_year = int(publication[0][1])
    last_pub_year = int(publication[-1][1])
    topic_dists = []
    topic_years = []
    for year in range(first_pub_year, last_pub_year+1, year_interval):
        filter_publ = [publ[0] for publ in publication if int(publ[1]) >= year and int(publ[1]) < year+2]
        #print(filter_publ)
        if len(filter_publ) > 0: 
            topic_dist  =  list(predict(topic_model, id2word, preprocess, filter_publ))
        else:
            topic_dist = [[]]
        topic_dists.append(topic_dist)
        topic_years.append(year)
    if len(topic_dists)==0:
        print(group_leader)
    return topic_years, topic_dists


# In[36]:


def aggregate_dist_over_interval(dist, i, rank=True):
    topic_dist_interval = []
    topic_rank = []
    for interval_topic in dist:
        temp_list=[]
        for topic in interval_topic:
            topic_dict = dict(topic)
            temp_list.append([topic_dict.get(i, 0) for i in range(25)])
        avg = np.mean(np.array(temp_list), axis=0)
        topic_dist_interval.append(avg)
        if rank:
            tmp_tuples = [(i, val)for i, val in enumerate(avg)]
            sort_tuples = sorted(tmp_tuples, key=lambda x: x[1], reverse=True)
            topic_rank.append(sort_tuples)#append([tup[0] for tup in sort_tuples])
    topic_dist_interval= np.array(topic_dist_interval)
    if len(topic_dist_interval) == 0:
        print(i)
    return (topic_dist_interval, topic_rank)


# In[37]:


def get_distance_over_interval(dist, i, func):
    metric_distance = []
    if len(dist) > 1:
        for p, q in zip(dist[0:-1],dist[1:]):
            dist_distance = func(p, q)
            metric_distance.append(dist_distance)
    else:
        dist_distance = func(dist[0])
        metric_distance.append(dist_distance)
    if len(metric_distance) == 0:
        print(i)
    return metric_distance


# In[38]:


#get_metrics_over_interval(formated_dist[16088])


# In[39]:


def kl_divergance(p, q=None, epsilon=0.000000001):
    p = p+epsilon
    if type(q)== np.ndarray or type(q)== list:
        q = q+epsilon
    dist = entropy(p, q)
    return dist


# In[40]:


# a= np.zeros(10)
# b= np.zeros(10)
# kl_divergance(a,b)


# In[41]:


def specific_plot_metric(mgp_id, year, metric_dist, researcher_id):
    index =  [i for i, idd in enumerate(researcher_id) if idd==mgp_id]
    if len(index) > 0:
        index=index[0]
        metric_dist = [[np.round(float(i), 2) for i in nested] for nested in metric_dist]
        plt.figure(figsize=(8, 6), dpi=80)
        print(index)
        if len(year[index][1:]) > 0:
            plt.plot(year[index][1:], metric_dist[index], label=f"MGP ID : {researcher_id[index]}",marker='o',markersize=5)
        else:
            plt.plot(year[index], metric_dist[index], label=f"MGP ID : {researcher_id[index]}",marker='o',markersize=5)
        plt.ylabel("Distance")#
        plt.xlabel("Publication Year")
        plt.legend()
        plt.show()
    else:
        print(f"Advisor (Group leader : {mgp_id}) not present in the filtered dataset")
    return


# In[42]:


def plot_metric(year, metric_dist, researcher_id, num_researcher=5):
    metric_dist = [[np.round(float(i), 2) for i in nested] for nested in metric_dist]
    plt.figure(figsize=(8, 6), dpi=80)
    for i in range(num_researcher):
        index = random.choice(range(0, len(year)))
        print(index)
        if len(year[index][1:]) > 0:
            plt.plot(year[index][1:], metric_dist[index], label=f"MGP ID : {researcher_id[index]}",marker='o',markersize=5)
        else:
            plt.plot(year[index], metric_dist[index], label=f"MGP ID : {researcher_id[index]}",marker='o',markersize=5)
    plt.ylabel("KL(P || Q)")
    plt.xlabel("Publication Year")
    plt.legend()
    plt.show()
    return


# In[43]:


def subplot_metric(year, metric_dist, researcher_id, num_researcher=5):
    metric_dist = [[np.round(float(i), 2) for i in nested] for nested in metric_dist]
    fig, axs = plt.subplots(2, num_researcher, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 0.5, wspace=0.2)
    axs = axs.ravel()
    for i in range(2*num_researcher):
        index = random.choice(range(0, len(year)))
        print(index)
        if len(year[index][1:]) > 0:
            year_mod  =  year[index][1:]
            axs[i].plot(year_mod, metric_dist[index], label=f"MGP ID : {researcher_id[index]}",marker='o',markersize=5)
        else:
            year_mod  =  year[index]
            axs[i].plot(year_mod, metric_dist[index], label=f"MGP ID : {researcher_id[index]}",marker='o',markersize=5)
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i].set_ylabel("Distance")#("KL(P || Q)")
        axs[i].set_xlabel("Publication Year")
        if len(year_mod) > 1 :
            year_range = list(range(min(year_mod), max(year_mod)))
        else:
            year_range = year_mod
        if len(year_range) > 10 and len(year_range) <= 40:
            year_range = year_range[0::5]
            add_year = year_range[-1]+5
            year_range.append(add_year)
        elif len(year_range) > 40 and len(year_range) <= 100:
            year_range = year_range[0::10]
            add_year = year_range[-1]+10
            year_range.append(add_year)
        elif len(year_range) <= 10 and len(year_range) > 1:
            year_range = year_range[0::2]
            add_year = year_range[-1]+2
            year_range.append(add_year)
        elif len(year_range) == 1:
            pass
            #year_range = year_range
        else:
            year_range = year_range[0::20]
            add_year = year_range[-1]+20
            year_range.append(add_year)
        axs[i].set_xticks(year_range)
        axs[i].legend()
    plt.show()
    return metric_dist


# In[44]:


def plot_devergance(metric_std):
    plt.plot(sorted(metric_std, reverse=True))
    plt.xlabel("Researchers")
    plt.ylabel("Standard Deviation")
    plt.show()
    return


# In[2]:


def modified_jaccard_index(topic_order1, topic_order2 = [], top=5):
    assert len(topic_order1) > 1 or len(topic_order2) > 1
    topic_order1 = topic_order1[:top]
    topic_order2 = topic_order2[:top]
    size1 =  len(topic_order1)
    size2 =  len(topic_order2)
    modified_order1 = [inx for i, inx in enumerate(topic_order1) for j in range(size1-(i))]
    modified_order2 = [inx for i, inx in enumerate(topic_order2) for j in range(size2-(i))]
    #print(modified_order1)
    #print(modified_order2)
    common_elements = list(set(modified_order1).intersection(modified_order2))
    distinct_elements = list(set(modified_order1).union(modified_order2))
    counter1 = Counter(modified_order1)
    counter2 = Counter(modified_order2)
    common_element_count = sum(min(counter1[elem], counter2[elem]) for elem in common_elements)
    #print(common_element_count)
    total_element_count =  sum(max(counter1[elem], counter2[elem]) for elem in distinct_elements)
    #print(total_element_count)
    mod_jaccard_index =  common_element_count/total_element_count
    dist = 1-mod_jaccard_index
    return dist


# In[46]:


def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    lags = range(2, max_lag)
    
    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    
    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    
    return reg[0]


# In[47]:


def find_top_lowest(metrics, group_head_id, not_zero=True,top=10):
    res_metric = [(head, metric) for head, metric in zip(group_head_id, metrics)]
    sorted_res_metric = sorted(res_metric, key= lambda x: x[1])
    #print(sorted_res_metric[0:5])
    if 
    res = [i[0] for i in sorted_res_metric] 
    persistent_res = res[:top] #if i[1] > 0
    anti_persistent_res = res[-top:] #if i[1] > 0
    return persistent_res, anti_persistent_res[::-1]


# In[48]:


def get_names(mgp_nodes, mgp_indices):
    details = [tuple(mgp_nodes[mgp_nodes["Id"]==index][["Id","Name","University", "Country"]].values[0]) for index in mgp_indices]
    return details


# In[ ]:


if __name__=="__main__":
    filtered_groups, filtered_grp_member_cy, mgpid2publ_count, group_publication_titles, mgp_nodes, mgp_edges= group_info()
    print(f"Total filtered groups: {len(filtered_groups)}")
    lda_model, id2word = load_topic_model()
    lda_model.minimum_probability =0.0
    group_head_id = [k for k,v in group_publication_titles.items() if len(v) > 0]
    topic_distribution_year = [(topic_distributions_over_interval((k,v), lda_model, id2word, prepare_text_for_lda)) for k,v in group_publication_titles.items() if len(v) > 0]
    interval_start_year = [i[0] for i in topic_distribution_year]
    topic_distribution = [i[1] for i in topic_distribution_year]
    aggregated_distribution_topic_rank = [aggregate_dist_over_interval(dist, i) for i, dist in enumerate(topic_distribution)]
    aggregated_distribution = [elem[0] for elem in aggregated_distribution_topic_rank]
    topic_with_value = [elem[1] for elem in aggregated_distribution_topic_rank]
    topic_rank = [[[value[0] for value in interval] for interval in researcher] for researcher in topic_with_value] #changed
    metric_distance_kl = [get_distance_over_interval(dist, i, kl_divergance) for i, dist in enumerate(aggregated_distribution)]
    metric_distance_tau = [get_distance_over_interval(rank, i, modified_jaccard_index) for i, rank in enumerate(topic_rank)]
    norm_distance_kl = [list(np.array(metrics)/sum(metrics)) if sum(metrics) > 0.0 else metrics for metrics in metric_distance_kl]
    norm_distance_tau = [list(np.array(metrics)/sum(metrics)) if sum(metrics) > 0.0 else metrics for metrics in metric_distance_tau]
    assert len(interval_start_year) == len(aggregated_distribution) == len(metric_distance_kl) ==len(metric_distance_tau) == len(group_head_id) == len(norm_distance_kl)==len(norm_distance_tau)
    print(f"Number of groups (Publication count > 0): {len(group_head_id)}")


# In[61]:


#mgp_nodes, mgp_edges = read_MGP()


# In[84]:


# # standard deviation
# metric_std_tau = [np.std(interval)/len(interval) for interval in norm_distance_tau]
# metric_peaks_tau = [len(find_peaks(interval)[0])/len(interval) for interval in norm_distance_tau]
# metric_hurst_exp_tau = [get_hurst_exponent(interval)/len(interval) for interval in norm_distance_tau]
# metric_hurst_exp_tau = [i for i in metric_hurst_exp_tau if ~np.isnan(i)]


# In[85]:


# # with Kullback–Leibler divergence (KL-Divergance)
# metric_std_kl = [np.std(interval)/len(interval) for interval in norm_distance_kl]
# metric_peaks_kl = [len(find_peaks(interval)[0])/len(interval) for interval in norm_distance_kl]
# metric_hurst_exp_kl = [get_hurst_exponent(interval)/len(interval) for interval in norm_distance_kl]
# metric_hurst_exp_kl = [i for i in metric_hurst_exp_kl if ~np.isnan(i)]


# In[86]:


# persistent_res, anti_persistent_res = find_top_lowest(metric_std_tau, group_head_id, top=10)


# In[87]:


# persistent_res


# In[88]:


# get_names(mgp_nodes, persistent_res)


# In[89]:


# get_names(mgp_nodes, anti_persistent_res)


# In[90]:


# anti_persistent_res


# In[91]:


# specific_plot_metric(284, interval_start_year, norm_distance_kl, group_head_id)


# In[92]:


# specific_plot_metric(151149, interval_start_year, norm_distance_kl, group_head_id)


# In[93]:


# group_publication_titles[151149]


# In[44]:


#topic_rank[10269]


# In[45]:


#norm_distance_tau[10269]


# In[94]:


# rounded_metric_dist = subplot_metric(interval_start_year, norm_distance_tau, group_head_id, 3)


# In[95]:


# rounded_metric_dist = subplot_metric(interval_start_year, norm_distance_kl, group_head_id, 3)


# In[96]:


# plot_metric(interval_start_year, norm_distance_tau, group_head_id, 5)


# In[97]:


# plot_metric(interval_start_year, norm_distance_kl, group_head_id, 5)


# In[98]:


# plot_devergance(sorted(metric_std_tau, reverse=True))


# In[99]:


# _= plt.hist(metric_std_tau, bins=100)
# plt.title("Deviation distribution")
# plt.show()


# In[100]:


# plt.plot(sorted(metric_peaks_tau,reverse=True))
# plt.ylabel("No. of peaks")
# plt.xlabel("Researchers index")
# plt.show()


# In[101]:


# _= plt.hist(metric_peaks_tau, bins=100)
# plt.title("Peak Distribution")
# plt.show()


# In[102]:


# plt.plot(sorted(metric_hurst_exp_tau, reverse=True))
# plt.ylabel("hurst exponent value")
# plt.xlabel("Researchers index")
# plt.show()


# In[103]:


# _= plt.hist(metric_hurst_exp_tau, bins=100)
# plt.title("Hurst exponent distribution")
# plt.show()


# In[189]:


# topic_order1 = [25,10, 15, 13 ,1]
# topic_order2 = []
# modified_jaccard_index(topic_order1, topic_order2)


# In[190]:


#from hurst import compute_Hc, random_walk
#metric_hurst_exp = [compute_Hc(group, kind='change', simplified=True) for group in norm_metric_distance]


# In[191]:


#print(year[2243],"\n",year[14984],"\n",year[13020],"\n",year[581],"\n",year[16228])


# In[193]:


#group_publ_zero = [k for k,v in group_publication_titles.items() if len(v) < 2] 
#number of groups with zero or one publication = 1299


# In[194]:


#len(group_publ_zero)


# In[195]:


#lda_model.show_topic(23)


# In[196]:


# list(predict(lda_model, id2word, prepare_text_for_lda, ['finit strain analysi elast theori', 'stress orthotrop elast layer']))


# In[ ]:


#mgp_edges[mgp_edges["advisor"]==210603]


# In[ ]:


#random.sample(list(filtered_groups.keys()),5)


# In[ ]:


#mgp_nodes["MSN"].apply(lambda x : x.rsplit("/")[-1] if type(x)==str else x)


# In[ ]:


#mathscinet2mgpid = dict(zip(mgp_nodes["mathscinet_id"], mgp_nodes["Id"]))


# In[ ]:


# for indx,i in enumerate(range(1939,2013,2)):
#     print(indx, i, i+2)


# In[ ]:


# data = (258, [('finit strain analysi elast theori', '1939'),
#  ('stress orthotrop elast layer', '1940'),
#  ('bend clamp plate', '1944'),
#  ('bend clamp plate', '1947'),
#  ('flow viscous fluid slowli rotat eccentr cylind', '1948'),
#  ('center flexur beam triangular section', '1949'),
#  ('involut conic orthogon matric', '1949'),
#  ('center flexur triangular beam', '1950'),
#  ('indent semi infinit medium axial symmetr rigid punch', '1952'),
#  ('solut dual integr equat plane strain boussinesq problem orthotrop medium',
#   '1953'),
#  ('common eigenvector commut oper', '1956'),
#  ('schwartz theori distribut', '1957'),
#  ('geometr note cayley transform', '1971'),
#  ('rotat matric', '1999'),
#  ('spatial decay cross diffus problem bound optim constant inequ ladyzhenskaya solonnikov',
#   '2007'),
#  ('blow phenomena nonlinear parabol system blow phenomena nonlinear parabol problem blow parabol problem robin boundari condit continu depend soret coeffici doubl diffus convect darci flow bound blow time nonlinear parabol problem spatial decay bound doubl diffus convect brinkman flow',
#   '2008'),
#  ('ill pose problem heat equat decay keller segel chemotaxi model bound blow time heat equat nonlinear boundari condit lower bound blow time nonlinear parabol problem',
#   '2009'),
#  ('blow phenomena semilinear heat equat nonlinear boundari condit blow phenomena semilinear heat equat nonlinear boundari condit ii blow decay criteria model chemotaxi',
#   '2010'),
#  ('blow class non linear parabol problem time depend coeffici robin type boundari condit blow phenomena parabol problem time depend coeffici neumann boundari condit blow phenomena class parabol system time depend coeffici lower bound blow model chemotaxi',
#   '2012'),
#  ('blow phenomena parabol problem time depend coeffici dirichlet boundari condit',
#   '2013')])


# In[ ]:





# In[ ]:


#publication = publicaton_data(stem_title=True)

