
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





<link rel="stylesheet" href="https://code.jquery.com/ui/1.10.4/themes/smoothness/jquery-ui.css">
<style>div.bk-hbox {
    display: flex;
    justify-content: center;
}

div.bk-hbox div.bk-plot {
    padding: 8px;
}

div.bk-hbox div.bk-data-table {
    padding: 20px;
}

div.hololayout {
  display: flex;
  align-items: center;
  margin: 0;
}

div.holoframe {
  width: 75%;
}

div.holowell {
  display: flex;
  align-items: center;
}

form.holoform {
  background-color: #fafafa;
  border-radius: 5px;
  overflow: hidden;
  padding-left: 0.8em;
  padding-right: 0.8em;
  padding-top: 0.4em;
  padding-bottom: 0.4em;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  margin-bottom: 20px;
  border: 1px solid #e3e3e3;
}

div.holowidgets {
  padding-right: 0;
  width: 25%;
}

div.holoslider {
  min-height: 0 !important;
  height: 0.8em;
  width: 100%;
}

div.holoformgroup {
  padding-top: 0.5em;
  margin-bottom: 0.5em;
}

div.hologroup {
  padding-left: 0;
  padding-right: 0.8em;
  width: 100%;
}

.holoselect {
  width: 92%;
  margin-left: 0;
  margin-right: 0;
}

.holotext {
  padding-left:  0.5em;
  padding-right: 0;
  width: 100%;
}

.holowidgets .ui-resizable-se {
  visibility: hidden
}

.holoframe > .ui-resizable-se {
  visibility: hidden
}

.holowidgets .ui-resizable-s {
  visibility: hidden
}


/* CSS rules for noUISlider based slider used by JupyterLab extension  */

.noUi-handle {
  width: 20px !important;
  height: 20px !important;
  left: -5px !important;
  top: -5px !important;
}

.noUi-handle:before, .noUi-handle:after {
  visibility: hidden;
  height: 0px;
}

.noUi-target {
  margin-left: 0.5em;
  margin-right: 0.5em;
}
</style>


<div class="logo-block">
<img src='data:image/png;base64,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
roTBs2RqAAAAAElFTkSuQmCC'
     style='height:25px; border-radius:12px; display: inline-block; float: left; vertical-align: middle'></img>


  <img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAjCAYAAAAe2bNZAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAK6wAACusBgosNWgAAABx0RVh0U29mdHdhcmUAQWRvYmUgRmlyZXdvcmtzIENTNui8sowAAAf9SURBVFiFvZh7cFTVHcc/59y7793sJiFAwkvAYDRqFWwdraLVlj61diRYsDjqCFbFKrYo0CltlSq1tLaC2GprGIriGwqjFu10OlrGv8RiK/IICYECSWBDkt3s695zTv9IAtlHeOn0O7Mzu797z+/3Ob/z+p0VfBq9doNFljuABwAXw2PcvGHt6bgwxhz7Ls4YZNVXxxANLENwE2D1W9PAGmAhszZ0/X9gll5yCbHoOirLzmaQs0F6F8QMZq1v/8xgNm7DYwwjgXJLYL4witQ16+sv/U9HdDmV4WrKw6B06cZC/RMrM4MZ7xz61DAbtzEXmAvUAX4pMOVecg9/MFFu3j3Gz7gQBLygS2RGumBkL0cubiFRsR3LzVBV1UMk3IrW73PT9C2lYOwhQB4ClhX1AuKpjLcV27oEjyUpNUJCg1CvcejykWTCXyQgzic2HIIBjg3pS6+uRLKAhumZvD4U+tq0jTrgkVKQQtLekfTtxIPAkhTNF6G7kZm7aPp6M9myKVQEoaYaIhEQYvD781DML/RfBGNZXAl4irJiwBa07e/y7cQnBaJghIX6ENl2GR/fGCBoz6cm5qeyEqQA5ZYA5x5eeiV0Qph4gjFAUSwAr6QllQgcxS/Jm25Cr2Tmpsk03XI9NfI31FTZBEOgVOk51adqDBNPCNPSRlkiDXbBEwOU2WxH+I7itQZ62g56OjM33suq1YsZHVtGZSUI2QdyYgkgOthQNIF7BIGDnRAJgJSgj69cUx1gB8PkOGwL4E1gPrM27gIg7NlGKLQApc7BmEnAxP5g/rw4YqBrCDB5xHkw5rdR/1qTrN/hKNo6YUwVDNpFsnjYS8RbidBPcPXFP6R6yfExuOXmN4A3jv1+8ZUwgY9D2OWjUZE6lO88jDwHI8ZixGiMKSeYTBamCoDk6kDAb6y1OcH1a6KpD/fZesoFw5FlIXAVCIiH4PxrV+p2npVDToTBmtjY8t1swh2V61E9KqWiyuPEjM8dbfxuvfa49Zayf9R136Wr8mBSf/T7bNteA8zwaGEUbFpckWwq95n59dUIywKl2fbOIS5e8bWSu0tJ1a5redAYfqkdjesodFajcgaVNWhXo1C9SrkN3Usmv3UMJrc6/DDwkwEntkEJLe67tSLhvyzK8rHDQWleve5CGk4VZEB1r+5bg2E2si+Y0QatDK6jUVkX5eg2YYlp++ZM+rfMNYamAj8Y7MAVWFqaR1f/t2xzU4IHjybBtthzuiAASqv7jTF7jOqDMAakFHgDNsFyP+FhwZHBmH9F7cutIYkQCylYYv1AZSqsn1/+bX51OMMjPSl2nAnM7hnjOx2v53YgNWAzHM9Q/9l0lQWPSCBSyokAtOBC1Rj+w/1Xs+STDp4/E5g7Rs2zm2+oeVd7PUuHKDf6A4r5EsPT5K3gfCnBXNUYnvGzb+KcCczYYWOnLpy4eOXuG2oec0PBN8XQQAnpvS35AvAykr56rWhPBiV4MvtceGLxk5Mr6A1O8IfK7rl7xJ0r9kyumuP4fa0lMqTBLJIAJqEf1J3qE92lMBndlyfRD2YBghHC4hlny7ASqCeWo5zaoDdIWfnIefNGTb9fC73QDfhyBUCNOxrGPSUBfPem9us253YTV+3mcBbdkUYfzmHiLqZbYdIGHHON2ZlemXouaJUOO6TqtdHEQuXYY8Yt+EbDgmlS6RdzkaDTv2P9A3gICiq93sWhb5mc5wVhuU3Y7m5hOc3So7qFT3SLgOXHb/cyOfMn7xROegoC/PTcn3v8gbKPgDopJFk3R/uBPWQiwQ+2/GJevRMObLUzqe/saJjQUQTTftEVMW9tWxPgAocwcj9abNcZe7s+6t2R2xXZG7zyYLp8Q1PiRBBHym5bYuXi8Qt+/LvGu9f/5YDAxABsaRNPH6Xr4D4Sk87a897SOy9v/fKwjoF2eQel95yDESGEF6gEMwKhLwKus3wOVjTtes7qzgLdXTMnNCNoEpbcrtNuq6N7Xh/+eqcbj94xQkp7mdKpW5XbtbR8Z26kgMCAf2UU5YEovRUVRHbu2b3vK1UdDFkDCyMRQxbpdv8nhKAGIa7QaQedzT07fFPny53R738JoVYBdVrnsNx9XZ9v33UeGO+AA2MMUkgqQ5UcdDLZSFeVgONnXeHqSAC5Ew1BXwko0D1Zct3dT1duOjS3MzZnEUJtBuoQAq3SGOLR4ekjn9NC5nVOaYXf9lETrUkmOJy3pOz8OKIb2A1cWhJCCEzOxU2mUPror+2/L3yyM3pkM7jTjr1nBOgkGeyQ7erxpdJsMAS9wb2F9rzMxNY1K2PMU0WtZV82VU8Wp6vbKJVo9Lx/+4cydORdxCCQ/kDGTZCWsRpLu7VD7bfKqL8V2orKTp/PtzaXy42jr6TwAuisi+7JolUG4wY+8vyrISCMtRrLKWpvjAOqx/QGhp0rjRo5xD3x98CWQuOQN8qumRMmI7jKZPUEpzNVZsj4Zbaq1to5tZZsKIydLWojhIXrJnES79EaOzv3du2NytKuxzJKAA6wF8xqEE8s2jo/1wd/khslQGxd81Zg62Bbp31XBH+iETt7Y3ELA0iU6iGDlQ5mexe0VEx4a3x8V1AaYwFJgTiwaOsDmeK2J8nMUOqsnB1A+dcA04ucCYt0urkjmflk9iT2v30q/gZn5rQPvor4n9Ou634PeBzoznes/iot/7WnClKoM/+zCIjH5kwT8ChQjTHPIPTjFV3PpU/Hx+DM/A9U3IXI4SPCYAAAAABJRU5ErkJggg=='
       style='height:15px; border-radius:12px; display: inline-block; float: left'></img>
  




</div>






```python
#something like that
#email_exchange = pd.read_json()
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




```{python echo=False}
# tupleizes the rows
# ghen is a holding dataframe which will be used again
ghen = pd_edges[['From', 'To', 'weight']]\
.groupby(['From', 'To'])\
.count()\
.reset_index()
bnbn = ghen.itertuples(index=False, name=None)
gma = []
for t in bnbn:
    gma.append(t)
```


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




![png](output_18_1.png)


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




![png](output_21_1.png)


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


```{python echo= False}
#Dictionaries to map Email Centrality ranks with
betw_dict = centrality_rank.set_index('Emails')['betweenness'].to_dict()
closeness_dict = centrality_rank.set_index('Emails')['closeness'].to_dict()
in_dict = centrality_rank.set_index('Emails')['in_degree'].to_dict()
out_dict = centrality_rank.set_index('Emails')['out_degree'].to_dict()
degree_dict = centrality_rank.set_index('Emails')['Degree_Centrality'].to_dict()
diff_degree_dict = centrality_rank.set_index('Emails')['Diff_Degree_Centrality'].to_dict()
median_dict = centrality_rank.set_index('Emails')['median_rank'].to_dict()
```


```{python echo= False}
# other centrality measures
#Betweenness
ghen['f_between'] = ghen['From'].apply(lambda x: betw_dict[x])
ghen['t_between'] = ghen['To'].apply(lambda x: betw_dict[x])
ghen['delta_between'] = ghen.t_between.subtract(ghen['f_between'])
# in_degree
ghen['f_in_degree'] = ghen['From'].apply(lambda x: in_dict[x])
ghen['t_in_degree'] = ghen['To'].apply(lambda x: in_dict[x])
ghen['delta_in_degree'] = ghen.t_in_degree.subtract(ghen['f_in_degree'])
# out_degree
ghen['f_out_degree'] = ghen['From'].apply(lambda x: out_dict[x])
ghen['t_out_degree'] = ghen['To'].apply(lambda x: out_dict[x])
ghen['delta_out_degree'] = ghen.t_out_degree.subtract(ghen['f_out_degree'])
# degree centrality
ghen['f_degree'] = ghen['From'].apply(lambda x: degree_dict[x])
ghen['t_degree'] = ghen['To'].apply(lambda x: degree_dict[x])
ghen['delta_degree'] = ghen.t_degree.subtract(ghen['f_degree'])
# degree centrality
ghen['f_diff_degree'] = ghen['From'].apply(lambda x: diff_degree_dict[x])
ghen['t_diff_degree'] = ghen['To'].apply(lambda x: diff_degree_dict[x])
ghen['delta_diff_degree'] = ghen.t_diff_degree.subtract(ghen['f_diff_degree'])
# closeness
ghen['f_closeness'] = ghen['From'].apply(lambda x: closeness_dict[x])
ghen['t_closeness'] = ghen['To'].apply(lambda x: closeness_dict[x])
ghen['delta_closeness'] = ghen.t_closeness.subtract(ghen['f_closeness'])
# closeness
ghen['f_median_rank'] = ghen['From'].apply(lambda x: median_dict[x])
ghen['t_median_rank'] = ghen['To'].apply(lambda x: median_dict[x])
ghen['delta_median_rank'] = ghen.t_median_rank.subtract(ghen['f_median_rank'])
```

```{python echo= False}
#functions to make graphing easier
def add_node_attributes(graph, node_dict, attr_name):
    """Add attributes to Multigraph from node_dictionary"""
    for nod in graph.nodes():
        graph.node[nod][attr_name] = node_dict[nod]

def frozen_nodes_dict(graph):
    """take a graph and return a dictionary of its nodes with only the username"""
    rename_dict = dict()
    for node in graph.nodes:
        rename_dict.update({node: next(iter(node))[:-10]})
        
    return rename_dict
```


```{python echo= False}
from holoviews.operation.datashader import datashade, bundle_graph #holoviews functions to make graphs easier to read


def enron_network_graph(graph, num_nodes, cent_indicator, dim_size, node_size, color_map):
    """Draws and interactive Network Graph displaying colors with a unique centrality indicator"""
    # makes a list of the emails ranked above num_nodes
    top_nodes = centrality_rank.loc[centrality_rank[cent_indicator]<=num_nodes]['Emails'].tolist()
        
    #creates a graph, adds position, creates a dict of the positions, and del the graph
    P = nx.DiGraph(graph).subgraph(top_nodes).copy()
    add_node_attributes(P, spring_pos, "Position")
    nx.relabel_nodes(P, frozen_nodes_dict(P), copy=False)
    pos = dict(P.nodes(data="Position"))
    del P
    #create the graph that will be used for the graph
    D = nx.DiGraph(graph).subgraph(top_nodes).copy()
    # add attribute to the graph
    cent_ind_dict = centrality_rank.set_index('Emails')[cent_indicator].to_dict()
    add_node_attributes(D, cent_ind_dict, cent_indicator)
    # takes relabels the nodes to the username without @enron.com
    nx.relabel_nodes(D, frozen_nodes_dict(D), copy=False)
    # creates the interative graph
    %%opts Graph [width=dim_size height=dim_size xaxis=None, yaxis=None colorbar=True]
    %%opts Graph (node_size=10)
    %%opts Graph (cmap=color_map node_size=node_size)
    %%opts Graph [tools=['hover']] (edge_hover_line_color='green' node_hover_fill_color='green')
    padding = dict(x=(-1.2, 1.2), y=(-1.2, 1.2)) 
    enron=hv.Graph.from_networkx(D, nx.spring_layout, pos = pos, k=2, iterations=50).redim.range(**padding).options(color_index=cent_indicator)
    #bundles the edges of the graph
    bndl = bundle_graph(enron, decay=0.1, initial_bandwidth=0.2, iterations=1)
    return bndl

def enron_network_graph_iterations(graph, num_nodes, cent_indicator, dim_size, node_size, color_map, iterations):
    """Draws and interactive Network Graph displaying colors with a unique centrality indicator"""
    # makes a list of the emails ranked above num_nodes
    top_nodes = centrality_rank.loc[centrality_rank[cent_indicator]<=num_nodes]['Emails'].tolist()
        
    #creates a graph, adds position, creates a dict of the positions, and del the graph
    P = nx.DiGraph(graph).subgraph(top_nodes).copy()
    add_node_attributes(P, spring_pos, "Position")
    nx.relabel_nodes(P, frozen_nodes_dict(P), copy=False)
    pos = dict(P.nodes(data="Position"))
    del P
    #create the graph that will be used for the graph
    D = nx.DiGraph(graph).subgraph(top_nodes).copy()
    # add attribute to the graph
    cent_ind_dict = centrality_rank.set_index('Emails')[cent_indicator].to_dict()
    add_node_attributes(D, cent_ind_dict, cent_indicator)
    # takes relabels the nodes to the username without @enron.com
    nx.relabel_nodes(D, frozen_nodes_dict(D), copy=False)
    # creates the interative graph
    %%opts Graph [width=dim_size height=dim_size xaxis=None, yaxis=None colorbar=True]
    %%opts Graph (node_size=10)
    %%opts Graph (cmap=color_map node_size=node_size)
    %%opts Graph [tools=['hover']] (edge_hover_line_color='green' node_hover_fill_color='green')
    padding = dict(x=(-1.2, 1.2), y=(-1.2, 1.2)) 
    enron=hv.Graph.from_networkx(D, nx.spring_layout, pos = pos, k=2, iterations=iterations).redim.range(**padding).options(color_index=cent_indicator)
    #bundles the edges of the graph
    bndl = bundle_graph(enron, decay=0.1, initial_bandwidth=0.2, iterations=1)
    return bndl

def get_graph(iteration):
    np.random.seed(10)
    return enron_network_graph_iterations(graph, num_nodes, cent_indicator, dim_size, node_size, color_map, iterations=iteration)
```


```{python echo= False}
num_nodes = 100
graph = G
dim_size=700
node_size = 35
color_map = 'Spectral'
diff_c = enron_network_graph(graph, num_nodes, 'Diff_Degree_Centrality', dim_size, node_size, color_map)
hv.renderer('bokeh').save(diff_c, 'diff_d_cent')
diff_c
```




<div id='b57f5d7e-e4ff-4044-a52c-c03710f08ede' style='display: table; margin: 0 auto;'>





  <div class="bk-root" id="e7725cf8-6c80-40dc-b581-03de2e16b5fa"></div>
</div>




```{python echo= False}
num_nodes = 100
graph = G
dim_size=700
node_size = 35
color_map = 'Spectral'
centsranklist = ['in_degree', 'out_degree', 'closeness', 'betweenness',
                 'Degree_Centrality', 'Diff_Degree_Centrality', 'median_rank']
graph_dict = {c:enron_network_graph(graph, num_nodes, c, dim_size, node_size, color_map)
              for c in centsranklist}
Ndlayout = hv.NdLayout(graph_dict, kdims="Centrality Indicator").cols(2)
CCcent = hv.HoloMap(Ndlayout)
hv.renderer('bokeh').save(CCcent, 'com_con_cent')
hv.HoloMap(Ndlayout)
```

<div class="hololayout row row-fluid">
  <div class="holoframe" id="display_area69a61376948a4b98887b5298e3948b88">
    <div id="_anim_img69a61376948a4b98887b5298e3948b88">
      
      <div id='240a8ccb-608a-49c6-befa-bca867dfd311' style='display: table; margin: 0 auto;'>





  <div class="bk-root" id="af12fda7-90bb-4c63-9a90-c28753676ac0"></div>
</div>
      
    </div>
  </div>
  <div class="holowidgets" id="widget_area69a61376948a4b98887b5298e3948b88">
    <form class="holoform well" id="form69a61376948a4b98887b5298e3948b88">
      
      
        <div class="form-group control-group holoformgroup" style=''>
          <label for="textInput69a61376948a4b98887b5298e3948b88_Centrality_Indicator"><strong>Centrality Indicator:</strong></label>
          <select class="holoselect form-control" id="_anim_widget69a61376948a4b98887b5298e3948b88_Centrality_Indicator" >
          </select>
        </div>
        
        
        </form>
    </div>
</div>




```python
num_nodes = 100
graph = G
dim_size=700
node_size = 35
color_map = 'Spectral'
cent_indicator = 'Diff_Degree_Centrality'
cent_iter = hv.HoloMap({i: get_graph(i)for i in range(2, 50, 3)}, kdims="Iterations")
hv.renderer('bokeh').save(cent_iter, 'cent_iter')
hv.HoloMap({i: get_graph(i)for i in range(2, 50, 3)}, kdims="Iterations")
```




<div class="hololayout row row-fluid">
  <div class="holoframe" id="display_areaf21edcf7beb04f55ac95c97b2daea1c4">
    <div id="_anim_imgf21edcf7beb04f55ac95c97b2daea1c4">
      
      <div id='96dc6521-e0db-4a60-99b8-7782ffdd6546' style='display: table; margin: 0 auto;'>





  <div class="bk-root" id="17a5f481-f6d4-47d5-856d-d02ee2418e19"></div>
</div>
      
    </div>
  </div>
  <div class="holowidgets" id="widget_areaf21edcf7beb04f55ac95c97b2daea1c4">
    <form class="holoform well" id="formf21edcf7beb04f55ac95c97b2daea1c4">
      
      
      <div class="form-group control-group holoformgroup" style=''>
        <label for="textInputf21edcf7beb04f55ac95c97b2daea1c4_Iterations">
          <strong>Iterations:</strong>
        </label>
        <div class="holowell">
          <div class="hologroup">
            <input type="text" class="holotext form-control input-small"
                   id="textInputf21edcf7beb04f55ac95c97b2daea1c4_Iterations" value="" readonly>
          </div>
          <div class="holoslider"
               id="_anim_widgetf21edcf7beb04f55ac95c97b2daea1c4_Iterations"></div>
        </div>
      </div>
      
        
        </form>
    </div>
</div>



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
ghen['FPS'] = edge_content['content'].apply(lambda x: words_tf(p1 , x))
ghen['SPS'] = edge_content['content'].apply(lambda x: words_tf(p2 , x))
ghen['TPS'] = edge_content['content'].apply(lambda x: words_tf(p3 , x))
ghen['its'] = edge_content['content'].apply(lambda x: words_tf(p4 , x))
ghen['FPP'] = edge_content['content'].apply(lambda x: words_tf(p5 , x))
ghen['TPP'] = edge_content['content'].apply(lambda x: words_tf(p6 , x))
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




![png](output_44_1.png)

### Top Nodes

```python
ghen = ghen.loc[ghen['f_median_rank']>150].loc[ghen['t_median_rank']>150]
```


```{python echo=False}
#spearman correlation as we are denoting a hierarchy centrality is monotonic
ghen_corr=ghen.corr('spearman')
sub_corr = ghen_corr[['FPS', 'SPS', 'TPS', 'its', 'FPP', 'TPP']]
sns.set()
f, ax = plt.subplots(figsize=(20, 20))
em_corr=sns.heatmap(sub_corr, annot=True, linewidths=.5, ax=ax, cmap='coolwarm')
em_corr.savefig("top_cent_corr.png")
em_corr
```

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



    C:\Users\benti\Anaconda3\envs\enron3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    




    <matplotlib.axes._subplots.AxesSubplot at 0x241f9746a90>




![png](output_52_2.png)



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




![png](output_53_1.png)


# Conclusion

These tests find little to no correlation between the term frequency of pronoun categories and social status within a hierarchy indicated by measures of centrality. The findings are evidence contrary to those in [Pennebaker's Paper](). A small correlation effect was found for delta degree difference centrality (r<sub>s</sub>=0.1) which measured the difference between in and out centrality. Although, this measure of centrality correctly ranked many executives the effect might also be due to the influence of number of correspondances and therefore the amount of pronouns generally.

The centrality measures above did rank top executives highly it was not clear that it continued to be predictive for email accounts with lower centrality. This might have played a role in the results.
