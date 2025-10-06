from AppFile import AppFile
from enums import CircuitBreaker
from enums import Actuation
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
import umap
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, mutual_info_score, adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

def PrintReducedDimensional(extractedFeatures, events, optimal_k, fileName):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(extractedFeatures)
    # tsne = TSNE(n_components=3, random_state=0)
    # umap_result = tsne.fit_transform(scaled_features)
    reducer = umap.UMAP(n_components=3, random_state=42)
    umap_result = reducer.fit_transform(scaled_features)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # cluster_labels = kmeans.predict(umap_result)
    # data = pd.DataFrame(umap_result)
    # sns.pairplot(data)

    # # 3D Plot
    signal_labels = [f"{events[i]}" for i in range(len(events))]
    fig = plt.figure(figsize=(8, 6))
    pio.renderers.default = "browser"
    fig = px.scatter_3d(
        x=umap_result[:, 0], 
        y=umap_result[:, 1],
        z=umap_result[:, 2],
        color=cluster_labels, 
        hover_name=signal_labels,  # Show labels on hover
        opacity=0.7,
        title=f"{fileName} - K-means Clusters"
    )
    pio.write_html(fig, file=f'{fileName}.html', auto_open=True)

    # plot_data = pd.DataFrame(umap_result)
    # plot_data['Cluster'] = cluster_labels  # Convert to string for categorical coloring
    # sns.pairplot(plot_data, hue="Cluster")
    # plt.show()

def FindOptimalKmeans(scaled_features):
    # -Find optimal K using Elbow Method-
    inertia = []
    silhouettes = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(scaled_features, kmeans.labels_))

    # Plot Elbow Curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouettes, 'go-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.tight_layout()
    plt.show()
    # -End- 

def PrintIMFCluster(extractedFeatures, events, characteristics):
    for i in range(7):
        fromItem = (i*characteristics)
        toItem = (i*characteristics) + characteristics
        items =  np.array([item[fromItem:toItem] for item in extractedFeatures])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(items)

        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        signal_labels = [f"{events[i]}" for i in range(len(events))]

        fig = plt.figure(figsize=(8, 6))
        pio.renderers.default = "browser"
        fig = px.scatter_3d(
            x=scaled_features[:, 0], 
            y=scaled_features[:, 1],
            z=scaled_features[:, 2],
            color=cluster_labels, 
            hover_name=signal_labels,  # Show labels on hover
            opacity=0.7,
            title="K-means Clusters"
        )
        pio.write_html(fig, file=f'3d_frequency_Mode_{i + 1}.html', auto_open=True)
    
def ParallelCoordinatesPlot(extractedFeatures, events):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(extractedFeatures)
    optimal_k = 4 
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)

    signal_labels = [f"{events[i]}" for i in range(len(events))]
    plot_data = pd.DataFrame(extractedFeatures)

    # Add cluster labels to your DataFrame
    plot_data['Cluster'] = cluster_labels  # Convert to string for categorical coloring

    # Create parallel coordinates plot colored by cluster
    fig = px.parallel_coordinates(
        plot_data,
        color="Cluster",  # Color by cluster label
        dimensions=[col for col in plot_data.columns if col != 'Cluster'],  # All feature columns
        color_continuous_scale=px.colors.qualitative.Plotly,  # Qualitative color scale for clusters
        title="Parallel Coordinates Plot with KMeans Clusters"
    )
    # Update layout for better readability
    fig.update_layout(
        margin=dict(l=50, r=50, b=50, t=50),
        width=3800,  # Wider to accommodate many features
        height=800
    )
    pio.renderers.default = "browser"
    pio.write_html(fig, file=f'Parallel_Coordinates_Tijuco.html', auto_open=True)

def ClusteringMetrics(extractedFeatures, optimal_k, sinalType):

    kmeans = KMeans(n_clusters=optimal_k)
    cluster_labels = kmeans.fit_predict(extractedFeatures)

    # Calculate clustering metrics
    silhouette = silhouette_score(extractedFeatures, cluster_labels)
    db_index = davies_bouldin_score(extractedFeatures, cluster_labels)
    ch_index = calinski_harabasz_score(extractedFeatures, cluster_labels)

    # Print the metric scores
    print(f"{sinalType} - K-Means:")
    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Davies-Bouldin Index: {db_index:.2f}")
    print(f"Calinski-Harabasz Index: {ch_index:.2f}")
    
    reducer = umap.UMAP(n_components=6, random_state=42)
    umap_result = reducer.fit_transform(extractedFeatures)
    cluster_labels = kmeans.fit_predict(umap_result)

    silhouette = silhouette_score(umap_result, cluster_labels)
    db_index = davies_bouldin_score(umap_result, cluster_labels)
    ch_index = calinski_harabasz_score(umap_result, cluster_labels)

    # Print the metric scores
    print(f"{sinalType} - UMAP K-Means:")
    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Davies-Bouldin Index: {db_index:.2f}")
    print(f"Calinski-Harabasz Index: {ch_index:.2f}")

def PrintTwoCurves(app: AppFile, tag):
    dados, samplingRate, tempos = app.obterSinaisManobra("C:\\Users\\jeffe\\PycharmProjects\\disjuntor\\File\\manobras\\Tijuco\\manobra_53_A\\", tag)
    plotQuantity = 2
    modes = plt.figure(1)
    plt.subplot(plotQuantity, 1, 1)
    plt.grid()
    plt.plot(tempos, dados, label='53_A')
    yLabel = "Acceleration (m/s²)"
    plt.ylabel(yLabel, fontsize=12, fontweight='bold')
    dados, samplingRate, tempos = app.obterSinaisManobra("C:\\Users\\jeffe\\PycharmProjects\\disjuntor\\File\\manobras\\Tijuco\\manobra_3_A\\", tag)
    plt.subplot(plotQuantity, 1, 2)
    plt.plot(tempos, dados, color='r', label=f'3_A')
    plt.xlabel('Time (ms)', fontsize=12, fontweight='bold')
    plt.ylabel(yLabel, fontsize=12, fontweight='bold')
    modes.tight_layout()
    modes.legend()
    plt.grid()
    modes.show()
    input()
    print('Fim')

if __name__ == "__main__":
    # Our parameters
    circuitBreaker = CircuitBreaker.Furnas
    actuation = Actuation.Closing
    tag = "M1VBSC"

    app = AppFile(circuitBreaker, actuation, tag)
    allIMFs, samplingRate, allIMFsFrequencyDomain, events, modes, n_samples, eventsWithDate = app.GetAllIMFsFromVMD()
    np.savetxt("eventsClosingDate.csv", eventsWithDate, fmt='%s', delimiter=',')
    for i in range(len(allIMFs)):
        np.savetxt(f"IMFs_Fechamento_Furnas\\IMFs - {events[i]}.csv", allIMFs[i], delimiter=",")
    for i in range(len(allIMFsFrequencyDomain)):
        np.savetxt(f"IMFs_Fechamento_Furnas\\IMFs Frequency - {events[i]}.csv", allIMFsFrequencyDomain[i], delimiter=",")

    # extractedFeatures, extractedFrequencyDomainFeatures, events = app.GetAllExtractedFeaturesFromVMD()
    # np.savetxt("extractedFeaturesOpening_Tijuco.csv", extractedFeatures, delimiter=",")
    # np.savetxt("extractedFrequencyDomainFeaturesOpening_Tijuco.csv", extractedFrequencyDomainFeatures, delimiter=",")
    # np.savetxt("eventsOpening_Tijuco.csv", events, fmt='%s', delimiter=',')

    # # -Time Domain-
    # extractedFeaturesClosing = pd.read_csv('extractedFeaturesClosing.csv', header=None).to_numpy()
    # extractedFeaturesOpening = pd.read_csv('extractedFeaturesOpening.csv', header=None).to_numpy()
    # extractedFeatures = np.concatenate((extractedFeaturesClosing, extractedFeaturesOpening))
    # # -End-

    # # -Frequency Domain-
    # extractedFrequencyDomainFeaturesClosing = pd.read_csv('extractedFrequencyDomainFeaturesClosing.csv', header=None).to_numpy()
    # extractedFrequencyDomainFeaturesOpening = pd.read_csv('extractedFrequencyDomainFeaturesOpening.csv', header=None).to_numpy()
    # extractedFrequencyDomainFeatures = np.concatenate((extractedFrequencyDomainFeaturesClosing, extractedFrequencyDomainFeaturesOpening))
    # # -End-

    # eventsOpening = pd.read_csv('eventsOpening.csv', header=None).to_numpy()
    # eventsClosing = pd.read_csv('eventsClosing.csv', header=None).to_numpy()
    # events = np.concatenate((eventsClosing, eventsOpening))

    # # -Time Domain-
    # extractedFeaturesClosing = pd.read_csv('extractedFeaturesClosing_Tijuco.csv', header=None).to_numpy()
    # extractedFeaturesOpening = pd.read_csv('extractedFeaturesOpening_Tijuco.csv', header=None).to_numpy()
    # extractedFeatures = np.concatenate((extractedFeaturesClosing, extractedFeaturesOpening))
    # # # -End-

    # # # -Frequency Domain-
    # extractedFrequencyDomainFeaturesClosing = pd.read_csv('extractedFrequencyDomainFeaturesClosing_Tijuco.csv', header=None).to_numpy()
    # extractedFrequencyDomainFeaturesOpening = pd.read_csv('extractedFrequencyDomainFeaturesOpening_Tijuco.csv', header=None).to_numpy()
    # extractedFrequencyDomainFeatures = np.concatenate((extractedFrequencyDomainFeaturesClosing, extractedFrequencyDomainFeaturesOpening))
    # # # -End-

    # eventsOpening = pd.read_csv('eventsOpening_Tijuco.csv', header=None).to_numpy()
    # eventsClosing = pd.read_csv('eventsClosing_Tijuco.csv', header=None).to_numpy()
    # events = np.concatenate((eventsClosing, eventsOpening))

    # PrintIMFCluster(extractedFrequencyDomainFeatures, events, 3)
    # ParallelCoordinatesPlot(extractedFeatures, events)
    # optimalK = 2
    # PrintReducedDimensional(extractedFeaturesOpening, eventsOpening, optimalK, "Sinais de Abertura do Contato - Tijuco")
    # ClusteringMetrics(extractedFeaturesClosing, optimalK)
    PrintTwoCurves(app, tag)