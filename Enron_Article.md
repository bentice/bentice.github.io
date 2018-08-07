---
layout: default
---
# Enron Network and Linguistics Analysis

Enron was an american energy company that went bankrupt in 2001. It was later revealed that the financial condition was sustained by institionalized, systematic, and creatively planned accounting fraud. The [Enron scandal](https://en.wikipedia.org/wiki/Enron_scandal), as it is now known, lead to the release of the Enron email corpus which was acquired by the Federal Energy Regulatory Commission during its investigation of the company's collapse. The corpus is unique as the only publicly available mass collections of real emails easily available for use in network analysis and machine learning.

In a series of studies Pennebaker found that lower status individuals used more first-person singular pronouns (I) compared with higher status individuals. In addition, he found that lower status individuals used first-person plural (we) more relative to lower status individuals. This was tested using various [centrality](https://en.wikipedia.org/wiki/Centrality) algorithms (degree, closeness, & betweenness) common in [Network Theory](https://en.wikipedia.org/wiki/Network_theory)  as a indicators of importance and status in a social network.


```python
import numpy as np
import email
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import holoviews as hv
import datashader as ds
hv.extension('bokeh')
```

```python
# creates a dataframe of edges and their weights
pd_edges = email_exchanges[['From', 'To']]
pd_edges['weight'] = 1
pd_edges[['From', 'To', 'weight']]\
.groupby(['From', 'To'])\
.count()\
.reset_index()\
.head(15)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>From</th>
      <th>To</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(tim.belden@enron.com)</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(brent.price@enron.com)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(steve.jackson@enron.com)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(jeffrey.gossett@enron.com)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(thomas.martin@enron.com)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(sally.beck@enron.com)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(susan.mara@enron.com)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(john.arnold@enron.com)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(matt.smith@enron.com)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(rebecca.cantrell@enron.com)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(scott.neal@enron.com)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(kevin.mcgowan@enron.com)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(ywang@enron.com)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(al.pollard@enron.com)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(paul.lucci@enron.com)</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
# creates a directed graph and adds weighted edges to it
G = nx.DiGraph()
G.add_weighted_edges_from(gma)
G.remove_edges_from(nx.selfloop_edges(G))
```

```python
# will create a basic visualization of the network in a spring layout
spring_pos = nx.spring_layout(G)
plt.figure(figsize=(10, 10))
plt.axis("off")
nx.draw_networkx(G,
                 pos= spring_pos,
                 with_labels=False,
                 node_size=1,
                 arrows=False,
                 widths=0.4)
```

![png](/assets/img/Enron/total_enron.png)


This layout was drawn with [Force-Directed Graph Algorithm](https://en.wikipedia.org/wiki/Force-directed_graph_drawing). These used ‘weight’ which corresponded to the number of emails sent between nodes. The more emails exchanged the closer nodes were brought together.

This basic drawing of the network is a birds-eye-view of the hierarchy. Executives and important employees would be at the center of the ring with many connections amongst each other and to the outer nodes.

## Hierarchy and Centrality Indicators

If centrality indicators are going to be a good measure of hierarchy within a network it would have a high number of Enron Executives rated as most central. Here are some executives to watch out for:

**Kenneth Lay** CEO and later Chairman

**Jeff Skilling** CEO

**Greg Whalley** President

**Sally.Beck** COO

**Steven Kean** Vice President and Chief of Staff

**Jeff Dasovich** Government Relation Executive

**John Lavorato** CEO of Enron America

**Louise Kitchen** President of Enron Online






**Degree Centrality** is the measurement of total nodes that the node is connected to vs total nodes in the network.

In Enron's case those that communicate often with more people compared to the number of people in the Enron Network overall. Degree Centrality has previously shown to be 83% accurate in predicting organizational dominance with the enron network using their titles to deduce the organizational hierarchy [(Agarwal, Omuya, Harnly, Rambow, 2012)](http://www.cs.columbia.edu/~apoorv/NewHomepage/Publications_files/aclShort2012.pdf). In degree centrality notes how many different nodes any given node is connected to based communications towards them. This is important as mass emails might not confer dominance but rather a communications role. We will look at in and out degree contrasted as well.


```python
incent = nx.in_degree_centrality(G) 
Algorithm = 'in_degree'
e_centrality = pd.DataFrame(list(incent.items()), columns=['Emails', Algorithm])
Email_Centrality = pd.DataFrame(list(incent.items()), columns=['Emails', Algorithm])
e_centrality.sort_values(Algorithm, ascending=False).head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emails</th>
      <th>in_degree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>443</th>
      <td>(louise.kitchen@enron.com)</td>
      <td>0.035435</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(sally.beck@enron.com)</td>
      <td>0.029468</td>
    </tr>
    <tr>
      <th>43</th>
      <td>(john.lavorato@enron.com)</td>
      <td>0.028361</td>
    </tr>
    <tr>
      <th>360</th>
      <td>(tana.jones@enron.com)</td>
      <td>0.026269</td>
    </tr>
    <tr>
      <th>361</th>
      <td>(sara.shackleton@enron.com)</td>
      <td>0.025715</td>
    </tr>
    <tr>
      <th>2288</th>
      <td>(kenneth.lay@enron.com)</td>
      <td>0.025161</td>
    </tr>
    <tr>
      <th>180</th>
      <td>(jeff.skilling@enron.com)</td>
      <td>0.022701</td>
    </tr>
    <tr>
      <th>392</th>
      <td>(greg.whalley@enron.com)</td>
      <td>0.022147</td>
    </tr>
    <tr>
      <th>515</th>
      <td>(mark.taylor@enron.com)</td>
      <td>0.021593</td>
    </tr>
    <tr>
      <th>444</th>
      <td>(vince.kaminski@enron.com)</td>
      <td>0.021470</td>
    </tr>
    <tr>
      <th>72</th>
      <td>(steven.kean@enron.com)</td>
      <td>0.021163</td>
    </tr>
    <tr>
      <th>81</th>
      <td>(jeff.dasovich@enron.com)</td>
      <td>0.020178</td>
    </tr>
    <tr>
      <th>125</th>
      <td>(gerald.nemec@enron.com)</td>
      <td>0.019317</td>
    </tr>
    <tr>
      <th>118</th>
      <td>(richard.shapiro@enron.com)</td>
      <td>0.019010</td>
    </tr>
    <tr>
      <th>200</th>
      <td>(rod.hayslett@enron.com)</td>
      <td>0.017964</td>
    </tr>
  </tbody>
</table>
</div>




```python
outcent = nx.out_degree_centrality(G)
Algorithm = 'out_degree'
e_centrality = pd.DataFrame(list(outcent.items()), columns=['Emails', Algorithm])
Email_Centrality = Email_Centrality.merge(e_centrality, on='Emails')
e_centrality.sort_values(Algorithm, ascending=False).head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emails</th>
      <th>out_degree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>81</th>
      <td>(jeff.dasovich@enron.com)</td>
      <td>0.038265</td>
    </tr>
    <tr>
      <th>360</th>
      <td>(tana.jones@enron.com)</td>
      <td>0.036235</td>
    </tr>
    <tr>
      <th>361</th>
      <td>(sara.shackleton@enron.com)</td>
      <td>0.031006</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(sally.beck@enron.com)</td>
      <td>0.029222</td>
    </tr>
    <tr>
      <th>444</th>
      <td>(vince.kaminski@enron.com)</td>
      <td>0.026515</td>
    </tr>
    <tr>
      <th>443</th>
      <td>(louise.kitchen@enron.com)</td>
      <td>0.025531</td>
    </tr>
    <tr>
      <th>1064</th>
      <td>(chris.germany@enron.com)</td>
      <td>0.023377</td>
    </tr>
    <tr>
      <th>150</th>
      <td>(kay.mann@enron.com)</td>
      <td>0.022455</td>
    </tr>
    <tr>
      <th>125</th>
      <td>(gerald.nemec@enron.com)</td>
      <td>0.021839</td>
    </tr>
    <tr>
      <th>108</th>
      <td>(susan.scott@enron.com)</td>
      <td>0.020855</td>
    </tr>
    <tr>
      <th>515</th>
      <td>(mark.taylor@enron.com)</td>
      <td>0.020424</td>
    </tr>
    <tr>
      <th>3742</th>
      <td>(michelle.cash@enron.com)</td>
      <td>0.020178</td>
    </tr>
    <tr>
      <th>5517</th>
      <td>(transportation.parking@enron.com)</td>
      <td>0.019932</td>
    </tr>
    <tr>
      <th>347</th>
      <td>(janette.elbertson@enron.com)</td>
      <td>0.019133</td>
    </tr>
    <tr>
      <th>503</th>
      <td>(d..steffes@enron.com)</td>
      <td>0.018210</td>
    </tr>
  </tbody>
</table>
</div>



**Closeness Centrality** This captures how close a node is to any given node in the network. We would expect executives to be high up on this measure.

For instance, low-level Enron employees in California will be very far from low-level employees in Europe. While executive management should have high closeness centrality because they can reach many nodes down the chain of command.


```python
close_cent = nx.closeness_centrality(G)
Algorithm = 'closeness'
e_centrality = pd.DataFrame(list(close_cent.items()), columns=['Emails', Algorithm])
Email_Centrality = Email_Centrality.merge(e_centrality, on='Emails')
e_centrality.sort_values(Algorithm, ascending=False).head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emails</th>
      <th>closeness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>443</th>
      <td>(louise.kitchen@enron.com)</td>
      <td>0.160202</td>
    </tr>
    <tr>
      <th>43</th>
      <td>(john.lavorato@enron.com)</td>
      <td>0.152906</td>
    </tr>
    <tr>
      <th>392</th>
      <td>(greg.whalley@enron.com)</td>
      <td>0.152086</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(sally.beck@enron.com)</td>
      <td>0.150729</td>
    </tr>
    <tr>
      <th>2288</th>
      <td>(kenneth.lay@enron.com)</td>
      <td>0.149302</td>
    </tr>
    <tr>
      <th>72</th>
      <td>(steven.kean@enron.com)</td>
      <td>0.148169</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(tim.belden@enron.com)</td>
      <td>0.146850</td>
    </tr>
    <tr>
      <th>444</th>
      <td>(vince.kaminski@enron.com)</td>
      <td>0.146255</td>
    </tr>
    <tr>
      <th>180</th>
      <td>(jeff.skilling@enron.com)</td>
      <td>0.145934</td>
    </tr>
    <tr>
      <th>49</th>
      <td>(david.delainey@enron.com)</td>
      <td>0.145595</td>
    </tr>
    <tr>
      <th>515</th>
      <td>(mark.taylor@enron.com)</td>
      <td>0.145426</td>
    </tr>
    <tr>
      <th>360</th>
      <td>(tana.jones@enron.com)</td>
      <td>0.143843</td>
    </tr>
    <tr>
      <th>1564</th>
      <td>(janet.dietrich@enron.com)</td>
      <td>0.143591</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(scott.neal@enron.com)</td>
      <td>0.143426</td>
    </tr>
    <tr>
      <th>274</th>
      <td>(m..presto@enron.com)</td>
      <td>0.143262</td>
    </tr>
  </tbody>
</table>
</div>



**Betweenness Centrality** Quantifies how many times a node acts as the shortest path between two nodes. 

For our purposes, betweenness centrality should show management roles. Managers will often be the shortest path to other management and other departments. Some employees will communicate with their managers more than they will communicate with people from other departments.


```python
betcent = nx.betweenness_centrality(G)
Algorithm = 'betweenness'
e_centrality = pd.DataFrame(list(betcent.items()), columns=['Emails', Algorithm])
Email_Centrality = Email_Centrality.merge(e_centrality, on='Emails')
e_centrality.sort_values(Algorithm, ascending=False).head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emails</th>
      <th>betweenness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>(sally.beck@enron.com)</td>
      <td>0.020167</td>
    </tr>
    <tr>
      <th>443</th>
      <td>(louise.kitchen@enron.com)</td>
      <td>0.018188</td>
    </tr>
    <tr>
      <th>81</th>
      <td>(jeff.dasovich@enron.com)</td>
      <td>0.018057</td>
    </tr>
    <tr>
      <th>444</th>
      <td>(vince.kaminski@enron.com)</td>
      <td>0.017002</td>
    </tr>
    <tr>
      <th>360</th>
      <td>(tana.jones@enron.com)</td>
      <td>0.016785</td>
    </tr>
    <tr>
      <th>361</th>
      <td>(sara.shackleton@enron.com)</td>
      <td>0.012619</td>
    </tr>
    <tr>
      <th>125</th>
      <td>(gerald.nemec@enron.com)</td>
      <td>0.011403</td>
    </tr>
    <tr>
      <th>2288</th>
      <td>(kenneth.lay@enron.com)</td>
      <td>0.010187</td>
    </tr>
    <tr>
      <th>43</th>
      <td>(john.lavorato@enron.com)</td>
      <td>0.010078</td>
    </tr>
    <tr>
      <th>271</th>
      <td>(shelley.corman@enron.com)</td>
      <td>0.009883</td>
    </tr>
    <tr>
      <th>72</th>
      <td>(steven.kean@enron.com)</td>
      <td>0.009833</td>
    </tr>
    <tr>
      <th>180</th>
      <td>(jeff.skilling@enron.com)</td>
      <td>0.009303</td>
    </tr>
    <tr>
      <th>3742</th>
      <td>(michelle.cash@enron.com)</td>
      <td>0.009082</td>
    </tr>
    <tr>
      <th>1064</th>
      <td>(chris.germany@enron.com)</td>
      <td>0.008882</td>
    </tr>
    <tr>
      <th>200</th>
      <td>(rod.hayslett@enron.com)</td>
      <td>0.008695</td>
    </tr>
  </tbody>
</table>
</div>



```python
Email_Centrality['Degree_Centrality'] = Email_Centrality[['in_degree', 'out_degree']].sum(axis=1)
Email_Centrality['Diff_Degree_Centrality'] = Email_Centrality['in_degree'].subtract(Email_Centrality['out_degree'])
```


```python
email_cent_corr = Email_Centrality.corr()
sns.set()
f, ax = plt.subplots(figsize=(20, 20))
c_corr = sns.heatmap(email_cent_corr, annot=True, linewidths=.5, ax=ax, cmap= 'coolwarm')
c_corr.savefig("c_corr.png")
c_corr
```

![png](/assets/img/Enron/c_corr.png)

In and out degree centrality were sharply contrasted in some key people like Jeff Skilling and Kenneth Lay. Because of this, I decided to make another measure of difference degree centrality. This was the only centrality indicator that put Skilling, Lay, Whalley, Lavorato, and Kitchen all at the top of the hierarchy. It is possible that executive and management positions receive communication from many sources but only send emails to a select few executives who delegate down the hierarchy. It is also possible that emails they sent were not included in the public data set for legal reasons.

In the heat map above we use the Pearson correlation and can see that most of the centrality measures are positively correlated. Closeness centrality less so with the others and difference of degree centrality has no correlation with the others except for a negative correlation with out degree which is expected.


```python
#Rank of Centrality 1 most central to N least central
centrality_rank = Email_Centrality\
.rank(axis='index', numeric_only=True, ascending=False)
centrality_rank['Emails'] = Email_Centrality['Emails']
centrality_rank = centrality_rank[['Emails', 'in_degree', 'out_degree', 'closeness',
                                   'betweenness', 'Degree_Centrality', 'Diff_Degree_Centrality']]
centrality_rank['median_rank'] = centrality_rank[['in_degree', 'out_degree', 'closeness',
                                   'betweenness', 'Degree_Centrality', 'Diff_Degree_Centrality']].median(axis=1)
```


```python
email_cent_corr = centrality_rank.corr('spearman')
sns.set()
f, ax = plt.subplots(figsize=(20, 20))
rank_corr = sns.heatmap(email_cent_corr, annot=True, linewidths=.5, ax=ax, cmap= 'coolwarm')
rank_corr.savefig("rank_corr.png")
rank_corr
```



![png](/assets/img/Enron/rank_corr.png)


In the heat map above we use the Spearman correlation and see that most of the centrality ranks are highly correlated. Naturally, difference degree centrality is negatively correlated because it is the in degree minus the out degree. The Spearman correlation is used because using centrality indicators as measures of hierarchy we expect them to be monotonic.

Below, ordered by median rank, we can see how many of the ranks differ. but many of the names of Enron executives that we were looking for are present. From looking at the most central emails we can see that many of our centrality indicators are a good measure of hierarchy within the social network.


```python
centrality_rank.sort_values(by=['median_rank'], ascending=[True]).head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Emails</th>
      <th>in_degree</th>
      <th>out_degree</th>
      <th>closeness</th>
      <th>betweenness</th>
      <th>Degree_Centrality</th>
      <th>Diff_Degree_Centrality</th>
      <th>median_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>443</th>
      <td>(louise.kitchen@enron.com)</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(sally.beck@enron.com)</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2478.0</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>360</th>
      <td>(tana.jones@enron.com)</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>12.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>16240.0</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>43</th>
      <td>(john.lavorato@enron.com)</td>
      <td>3.0</td>
      <td>20.5</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>5.50</td>
    </tr>
    <tr>
      <th>361</th>
      <td>(sara.shackleton@enron.com)</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>29.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>16175.5</td>
      <td>5.50</td>
    </tr>
    <tr>
      <th>444</th>
      <td>(vince.kaminski@enron.com)</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>16161.0</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>2288</th>
      <td>(kenneth.lay@enron.com)</td>
      <td>6.0</td>
      <td>1781.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>26.0</td>
      <td>1.0</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>81</th>
      <td>(jeff.dasovich@enron.com)</td>
      <td>12.0</td>
      <td>1.0</td>
      <td>37.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>16255.0</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>180</th>
      <td>(jeff.skilling@enron.com)</td>
      <td>7.0</td>
      <td>751.5</td>
      <td>9.0</td>
      <td>12.0</td>
      <td>29.0</td>
      <td>2.0</td>
      <td>10.50</td>
    </tr>
    <tr>
      <th>125</th>
      <td>(gerald.nemec@enron.com)</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>87.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>15924.0</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>72</th>
      <td>(steven.kean@enron.com)</td>
      <td>11.0</td>
      <td>17.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>41.5</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>515</th>
      <td>(mark.taylor@enron.com)</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>17.0</td>
      <td>8.0</td>
      <td>454.0</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>392</th>
      <td>(greg.whalley@enron.com)</td>
      <td>8.0</td>
      <td>240.0</td>
      <td>3.0</td>
      <td>39.0</td>
      <td>22.5</td>
      <td>3.0</td>
      <td>15.25</td>
    </tr>
    <tr>
      <th>3742</th>
      <td>(michelle.cash@enron.com)</td>
      <td>24.0</td>
      <td>12.0</td>
      <td>28.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>16218.0</td>
      <td>18.50</td>
    </tr>
    <tr>
      <th>1064</th>
      <td>(chris.germany@enron.com)</td>
      <td>25.0</td>
      <td>7.0</td>
      <td>179.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>16247.0</td>
      <td>19.50</td>
    </tr>
  </tbody>
</table>
</div>


<iframe src="/assets/img/Enron/com_con_cent.html"
        sandbox="allow-same-origin allow-scripts"
        width = "1000"
        height="700" 
        scrolling="no"
        seamless="seamless"
        frameborder="0">
</iframe>

This is an interactive map showing the most central node and their arrangement in a force directed graph. Dark red are the most important nodes ranked. Ideally our highly ranked nodes would show up in the center of our graph.

Below difference degree centrality is a good example as nodes in the center are closer to dark red.

<iframe src="/assets/img/Enron/diff_d_cent.html"
        sandbox="allow-same-origin allow-scripts"
        width = "700"
        height="700" 
        scrolling="no"
        seamless="seamless"
        frameborder="0">
</iframe>

Here is a look at the force directed graph working through the iterations from the inital point that these nodes had when using the whole network. They are colored using difference degree centrality.

<iframe src="/assets/img/Enron/cent_iter.html"
        sandbox="allow-same-origin allow-scripts"
        width = "1000"
        height="700"  
        scrolling="no"
        seamless="seamless"
        frameborder="0">
</iframe>


## Term Frequency of Pronouns

We want to measure the term frequency of pronouns and compare them to the centrality difference between the Recipient (To) and the Sender (From). We expect that C<sub>Delta</sub> centrality will have a negative correlation with use of First Person Singular (FPS) pronouns as emails travel down the hierarchy (for instance for people ranked 1 emailing someone ranked 30 C<sub>Delta</sub> = 29) and a positive correlation with the use of First Person Plural (FPP) pronouns for emails traveling down the hierarchy.

Let
C<sub>Delta</sub> = C<sup>rank</sup><sub>To</sub> - C<sup>rank</sup><sub>From</sub>


then as C<sub>Delta</sub>->-infty will TF(FPS)->0 and as C<sub>Delta</sub>->infty will TF(FPP)->1

Our hypotheses:

H<sub>null</sub>: Corr(C<sub>Delta</sub>, TF(FPS)) = 0 and Corr(C<sub>Delta</sub>, TF(FPP)) = 0


H<sub>alternative</sub>: Corr(C<sub>Delta</sub>, TF(FPS)) <= -0.1 and/or Corr(C<sub>Delta</sub>, TF(FPP)) >= 0.1

```python
edge_content = email_exchanges.groupby(['From', 'To'])['content'].apply(lambda x: x.str.cat(sep=' ')).reset_index()
edge_content.head(10) #concatination of all emails between two nodes
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>From</th>
      <th>To</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(tim.belden@enron.com)</td>
      <td>Here is our forecast Tim  Matt sent you a emai...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(brent.price@enron.com)</td>
      <td>Will   Here is a list of the top items we need...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(steve.jackson@enron.com)</td>
      <td>Will   Here is a list of the top items we need...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(jeffrey.gossett@enron.com)</td>
      <td>Susan hours are out of hand  We need to find a...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(thomas.martin@enron.com)</td>
      <td>attached is the systems wish list for the gas ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(sally.beck@enron.com)</td>
      <td>Susan hours are out of hand  We need to find a...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(susan.mara@enron.com)</td>
      <td>Attached  are two files that illustrate the fo...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(john.arnold@enron.com)</td>
      <td>attached is the systems wish list for the gas ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(matt.smith@enron.com)</td>
      <td>Can you guys coordinate to make sure someone l...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(rebecca.cantrell@enron.com)</td>
      <td>Attached  are two files that illustrate the fo...</td>
    </tr>
  </tbody>
</table>
</div>

**FPS** First Person Singular

**SPS** Second Person Singular

**TPS** Third Person Singular

**its** it its itself

**FPP** First Person Plural

**TPP** Third Person Plural


```python
def words_tf(words, string):
    "finds the term frequency of a given list of pronouns"
    text = nltk.word_tokenize(string)
    g = 0
    for word in words:
        g+=float(text.count(word))
        if len(text)>0:
            return g / len(text)
        else:
            return 0
```

```python
#spearman correlation as we are denoting a hierarchy centrality is monotonic
ghen_corr=ghen.corr('spearman')
sub_corr = ghen_corr[['FPS', 'SPS', 'TPS', 'its', 'FPP', 'TPP']]
sns.set()
f, ax = plt.subplots(figsize=(20, 20))
em_corr=sns.heatmap(sub_corr, annot=True, linewidths=.5, ax=ax, cmap='coolwarm')
em_corr.savefig("all_email_corr.png")
em_corr
```

![png](/assets/img/Enron/all_email_corr.png)

The correlation matix above does not find an effect above our + or - 0.1 threshold between various delta centralities and pronoun types.

There are correlations between the from centrality and pronouns types for instance highly ranked employees by in_degree were significantly less likely to use First Person Singular (FPS) pronouns (r<sub>s</sub>=-0.13) and also less likely to use First Person Plural (FPP) pronouns (r<sub>s</sub>=-0.11). This is interesting as it is at least partial evidence for Pennebakers hypothesis that high-status individuals use less FPS pronouns than low-status individuals. It is also evidence against his finding that high-status individuals use more FPP pronouns.

Difference degree centrality of from nodes also found that higher-status individuals were more likely to use Second Person Singular (SPS) pronouns (r<sub>s</sub>=0.12). This means that when sending emails these people were more likely to be using imperative sentences. Difference degree centrality was a measure of the difference between in and out connected nodes which means these individuals had many different people email them than they emailed themselves. This was a good indication of the top two executives Skilling and Lay. The correlation could signal that when these nodes sent emails were more likely to be commands.

### One to One Correspondences (N recipients=1)

Emails sent to more than one recipient may cause senders to use pronouns differently. Below we do the same analysis correspondents that include only one to one correspondences.


```python
# edge content with 1 recipient
edge_content = email_exchanges.loc[email_exchanges['n_recipients']<2].groupby(['From', 'To'])['content'].apply(lambda x: x.str.cat(sep=' ')).reset_index()
edge_content.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>From</th>
      <th>To</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(tim.belden@enron.com)</td>
      <td>Here is our forecast Tim  Matt sent you a emai...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(chad.landry@enron.com)</td>
      <td>Chad  Call Ted Bland about the trading track p...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(karen.buckley@enron.com)</td>
      <td>I think Chad deserves an interview The topic w...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(andrea.richards@enron.com)</td>
      <td>Send his resume to Karen Buckley  I believe th...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(naomi.johnston@enron.com)</td>
      <td>Naomi  The two analysts that I have had contac...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(mary.gray@enron.com)</td>
      <td>Griff  It is bidweek again  I need to provide ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(brenda.flores-cuellar@enron.com)</td>
      <td>JeffBrenda  Please authorize the following pro...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(slewis2@enron.com)</td>
      <td>Susan   I received an enrollment confirmation ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(candace.womack@enron.com)</td>
      <td>vishal resigned today</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(phillip.allen@enron.com)</td>
      <td>(julie.gomez@enron.com)</td>
      <td>here is the file I showed you Julie   The numb...</td>
    </tr>
  </tbody>
</table>
</div>


```python
#Spearman Corr
ghen_corr=ghen.corr('spearman')
sub_corr = ghen_corr[['FPS', 'SPS', 'TPS', 'its', 'FPP', 'TPP']]
sns.set()
f, ax = plt.subplots(figsize=(20, 20))
em_corr=sns.heatmap(sub_corr, annot=True, linewidths=.5, ax=ax, cmap='coolwarm')
em_corr.savefig("private_email_corr.png")
em_corr
```


![png](/assets/img/Enron/private_email_corr.png)

There were no findings here to indicate a relationship between centrality and pronoun usage. This could because one to one emails were more likely to happend between equals and so power dynamics were less of a factor.

# Conclusion

These tests find little to no correlation between the term frequency of pronoun categories and social status dynamic between the sender and reciever of emails. Some of the findings do confirm some of Pennebaker's findings. Notably, in degree centrality and difference degree centrality showed evidence of power dynamics at play in the intra-Enron social network. High-status individuals measured by in degree were less likely to use first person singular pronouns as well as first person plural pronouns. High-status difference degree were measured to use more second person pronouns perhaps indicating more imperitive sentences. The findings both support and go contrary to those in [Pennebaker's Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.904.6689&rep=rep1&type=pdf).

The dynamic that Pennebaker proposed did not manifest itself in communication between nodes. The corrleation between pronoun usage and the difference of centrality were close to zero (-0.1>r<sub>s</sub><0.1)

The centrality measures above did rank top executives highly but it was not clear that it continued to be predictive for email accounts with lower centrality. The centrality measures might not capture nodes of equal rank. After the top eschelon of Enron nodes would start to have many more equals than subordinates or superiours and would communicate on this basis. The centrality rank would not reflect this.

Using the communications of a social network to map the social hierarchy is problematic. Some roles may require a dispropotionate representation in the communication network. Secretaries of Executives might rank higher than upper level management because of their position as an intermediary. These caveats should be taken into account when considering the findings.

The full notebook including data cleaning and preprocessing analysis can be viewed [here](Enron_Data_Prep.slides.html) and the full network analysis can be viewed [here](Enron_Article.slides.html).
