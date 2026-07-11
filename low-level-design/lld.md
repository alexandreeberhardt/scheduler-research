# Contraintes fonctionnelles et non fonctionnelles.

Le modèle doit tourner très vite, dans un temps suffisament court pour ne pas introduire de latence notable.
Le CPU doit fournir les tâches concurentes candidates à chaque instant.
La communication entre les deux ne doit pas introduire de latence notable.
La communication ne doit pas être plus couteuse que les taches ordonancées.

# Scénarios d’utilisation

Le CPU a plusieurs taches concurentes. Il envoie la liste des taches en input à un modèle d'IA responsable de la selection de la tache à executer (qui tourne probablement sur un GPU/cluster, local ou non). Ensuite le modèle d'IA indique la tache suivante (OU **LES** TACHES SUIVANTES ??) puis le CPU l'execure et recommence le cycle.

# Responsabilités principales

CPU : envoyer les taches du contexte, reçevoir les taches séléctionnées, et executer les taches selectionnées.
GPU/CLUSTER/Hardware : reçevoir la liste des taches et inférer le modèle d'IA pour choisir la tache puis envoyer les taches séléctionnées.


