---
layout: default
---

# Enron Email Corpus: Network and Linguistic Analysis of Hierarchy

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
#test['From'] = test['From'].map(lambda x: next(iter(x))[:-10])
#test['To'] = test['To'].map(lambda x: next(iter(x))[:-10])
pd_edges['weight'] = 1
pd_edges[['From', 'To', 'weight']]\
.groupby(['From', 'To'])\
.count()\
.reset_index()\
.head(15)
```

    C:\Users\benti\Anaconda3\envs\enron3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    




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
#add some output
```

    C:\Users\benti\Anaconda3\envs\enron3\lib\site-packages\networkx\drawing\layout.py:499: RuntimeWarning: invalid value encountered in sqrt
      distance = np.sqrt((delta**2).sum(axis=0))
    


![png](output_6_1.png)


This layout was drawn wth [Force-Directed Graph Algorithm](https://en.wikipedia.org/wiki/Force-directed_graph_drawing).These used 'weight' which corresponed to the number of emails sent between nodes. The more emails exchanged the closer nodes were brought together.

## Hierarchy and Centrality Indicators

If Centrality indicators are going to be a good measure of hierarchy within a network it would have a high numuber of Enron Executives rated as most Central. Here are some to watch out for:

**Kenneth Lay** CEO and later Chairman

**Jeff Skilling** CEO

**Greg Whalley** President

**Sally.Beck** COO

**Steven Kean** Vice President and Chief of Staff

**Jeff Dasovich** Government Relation Executive

**John Lavorato** CEO of Enron America

**Louise Kitchen** President of Enron Online




**Degree Centrality** is the measurement of total nodes that is connected to vs total nodes in the network.

In Enrons case those that communicate often with more people compared to the number of people in the Enron Network overall. Degree Centrality has previously shown to be 83% accurate in predicting organizational dominance pair using the their titles to deduce the organizational hierarchy [(Agarwal, Omuya, Harnly, Rambow, 2012)](http://www.cs.columbia.edu/~apoorv/NewHomepage/Publications_files/aclShort2012.pdf). In degree centrality notes how many different nodes any given node is connected to based communications towards them. This is important as mass emails might not confer dominance but rather a communications role. We will look at in and out degree seperately as well as together.


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

For instance low level enron employees in California will be very far from Low level employees in Europe. While executive management should have high closeness centrality because they can reach many nodes down the chain of command.


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

For our purposes betweenness centrality should show Management Roles. Managers will often be the shortest path to other management and other departments. Some employees will communicate with their managers more than they will communicate with people from other departments.


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




    <matplotlib.axes._subplots.AxesSubplot at 0x241c2c76780>




![png](/assets/img/Enron/c_corr.png)


In the heatmap above we use the pearson correlation and can see that most of the centrality measures are positively correlated. Closeness Centrality less so with the others and Difference of Degree Centrality has no correlation with the others except for a negative correlation with out degree which is expected.


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




    <matplotlib.axes._subplots.AxesSubplot at 0x241bc6b7550>




![png](/assets/img/Enron/rank_corr.png)


In the heatmap above we use the spearman correlation and see that most of the centrality ranks are positively correlated. The spearman correlation is used because the assumption of normality is violated and using centrality indicators as measures of hierarchy we expect them to be monotonic.


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



In and out degree centrality were sharply contrasted in some key people like Jeff Skilling and Kenneth Lay. Because of this I decided to make another measure of difference degree centrality. This was the only centrality indicator that put Skilling, Lay, Whalley, Lavorato, and Kitchen all at the top of the hierarchy. It is possible that executive and management positions recieve communication from many sources but only send emails to a select few executives who delegate down the hierarchy. It is also possible that Emails they sent were not included in the public data set for legal reasons.



<iframe src="/assets/img/Enron/diff_d_cent.html"
        sandbox="allow-same-origin allow-scripts"
        width = "700"
        height="700" 
        scrolling="no"
        seamless="seamless"
        frameborder="0">
</iframe>


<iframe src="/assets/img/Enron/com_con_cent.html"
        sandbox="allow-same-origin allow-scripts"
        width = "1000"
        height="700" 
        scrolling="no"
        seamless="seamless"
        frameborder="0">
</iframe>

<iframe src="/assets/img/Enron/cent_iter.html"
        sandbox="allow-same-origin allow-scripts"
        width = "1000"
        height="700"  
        scrolling="no"
        seamless="seamless"
        frameborder="0">
</iframe>


## Term Frequency of Pronouns

We want to measure the term frequency of pronouns and compare them to the centrality difference between the Recipient (To) and the Sender (From). We expect that C<sub>$\Delta$</sub> centrality will have a negative correlation with use of First Person Singular (FPS) pronouns as emails travel down the hierarchy (For instance for people ranked 1 emailing someone ranked 30 C<sub>$\Delta$</sub> = 29) and a positive correlation with the use of First Person Plural (FPP) pronouns for emails traveling down the hierarchy.

C<sub>$\Delta$</sub>=C<sup>rank</sup><sub>To</sub> - C<sup>rank</sup><sub>From</sub>
then as C<sub>$\Delta$</sub>->$-\infty$ will TF(FPS)->0 and as C<sub>$\Delta$</sub>->$\infty$ will TF(FPP)->1

H<sub>null</sub>: Corr(C<sub>$\Delta$</sub>, TF(FPS)) = 0 and Corr(C<sub>$\Delta$</sub>, TF(FPP)) = 0

H<sub>alternative</sub>: Corr(C<sub>$\Delta$</sub>, TF(FPS)) <= -0.1 and/or Corr(C<sub>$\Delta$</sub>, TF(FPP)) >= 0.1

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




    <matplotlib.axes._subplots.AxesSubplot at 0x241f61c7898>




![png](/assets/img/Enron/all_email_corr.png)


### Private Emails (n_recipients=1)

Emails sent to more than one recipient may cause senders to use pronouns differently. Below we do the same analysis correspondents that include only two people


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




    <matplotlib.axes._subplots.AxesSubplot at 0x241f74d27f0>




![png](/assets/img/Enron/private_email_corr.png)


# Conclusion

These tests find little to no correlation between the term frequency of pronoun categories and social status within a hierarchy indicated by measures of centrality. The findings are evidence contrary to those in [Pennebaker's Paper](). A small correlation effect was found for delta degree difference centrality (r<sub>s</sub>=0.1) which measured the difference between in and out centrality. Although, this measure of centrality correctly ranked many executives the effect might also be due to the influence of number of correspondances and therefore the amount of pronouns generally.

The centrality measures above did rank top executives highly it was not clear that it continued to be predictive for email accounts with lower centrality. This might have played a role in the results.
