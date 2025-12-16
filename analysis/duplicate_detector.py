"""
Duplicate Detector Module
=========================

Modul za detekciju copy-paste novinarstva i praćenje širenja vijesti.

Koristi sentence embeddings za semantic similarity detection:
- paraphrase-multilingual-MiniLM-L12-v2 (multilingual za hrvatski)
- Cosine similarity za mjerenje sličnosti
- Vremensko praćenje - tko je prvi objavio

Thresholds:
- >= 0.85: Duplikat (copy-paste)
- 0.70-0.85: Jako slično (parafrazirano)
- < 0.70: Originalno/različito

Author: News Trend Analysis Team
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DuplicateDetector:
    """
    Detektor duplikata temeljen na semantic similarity.
    
    Koristi sentence-transformers za generiranje embeddings i 
    cosine similarity za određivanje sličnosti članaka.
    
    Attributes:
        model_name: Ime sentence transformer modela
        threshold_duplicate: Prag za duplikat (default: 0.85)
        threshold_similar: Prag za slično (default: 0.70)
        model: SentenceTransformer instanca
        
    Example:
        >>> detector = DuplicateDetector()
        >>> results = detector.find_duplicates(articles_df)
        >>> analysis = detector.get_spread_analysis(results)
    """
    
    def __init__(
        self,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        threshold_duplicate: float = 0.85,
        threshold_similar: float = 0.70,
        batch_size: int = 32
    ):
        """
        Inicijaliziraj detector.
        
        Args:
            model_name: Sentence transformer model (multilingual za hrvatski)
            threshold_duplicate: Prag za označavanje duplikata (0.85 = 85% sličnosti)
            threshold_similar: Prag za označavanje sličnog sadržaja
            batch_size: Batch size za encoding
        """
        self.model_name = model_name
        self.threshold_duplicate = threshold_duplicate
        self.threshold_similar = threshold_similar
        self.batch_size = batch_size
        
        # Lazy loading modela
        self._model = None
        
        logger.info(f"DuplicateDetector initialized (duplicate_threshold={threshold_duplicate}, "
                   f"similar_threshold={threshold_similar})")
    
    @property
    def model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device='cpu')
                logger.info("✅ Model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def compute_embeddings(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Izračunaj embeddings za listu tekstova.
        
        Args:
            texts: Lista tekstova
            show_progress: Prikaži progress bar
            
        Returns:
            Numpy array embeddings [n_texts, embedding_dim]
        """
        logger.info(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Izračunaj matricu cosine similarity.
        
        Args:
            embeddings: Embedding matrica [n_samples, embedding_dim]
            
        Returns:
            Similarity matrica [n_samples, n_samples]
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings)
    
    def find_duplicates(
        self, 
        df: pd.DataFrame,
        title_col: str = 'title',
        content_col: str = 'text',
        date_col: str = 'publishedAt',
        source_col: str = 'source',
        use_title_only: bool = False,
        max_content_chars: int = 500
    ) -> pd.DataFrame:
        """
        Pronađi duplikate i označi originale.
        
        Args:
            df: DataFrame s člancima
            title_col: Ime kolone s naslovom
            content_col: Ime kolone sa sadržajem
            date_col: Ime kolone s datumom
            source_col: Ime kolone s izvorom
            use_title_only: Koristi samo naslov za embedding
            max_content_chars: Max znakova sadržaja za embedding
            
        Returns:
            DataFrame s dodanim kolonama:
            - is_original: bool - je li originalni članak
            - duplicate_of: int/None - index originalnog članka
            - similarity_score: float - max sličnost s ranijim člankom
            - original_source: str - portal koji je prvi objavio
            - similarity_category: str - 'original', 'duplicate', 'similar'
        """
        logger.info(f"Finding duplicates in {len(df)} articles...")
        
        # Kopiraj DataFrame
        result_df = df.copy()
        
        # Parsiraj datume i sortiraj po vremenu
        result_df[date_col] = pd.to_datetime(result_df[date_col], errors='coerce')
        result_df = result_df.sort_values(date_col).reset_index(drop=True)
        
        # Pripremi tekst za embedding
        if use_title_only:
            texts = result_df[title_col].fillna('').tolist()
        else:
            texts = [
                f"{row[title_col]} {str(row[content_col])[:max_content_chars]}"
                for _, row in result_df.iterrows()
            ]
        
        # Izračunaj embeddings
        embeddings = self.compute_embeddings(texts)
        
        # Inicijaliziraj rezultatne kolone
        result_df['is_original'] = True
        result_df['duplicate_of'] = None
        result_df['similarity_score'] = 0.0
        result_df['original_source'] = result_df[source_col]
        result_df['similarity_category'] = 'original'
        
        # Pronađi duplikate (usporedi svaki članak s PRETHODNIM člancima)
        n_duplicates = 0
        n_similar = 0
        
        for i in range(1, len(result_df)):
            # Cosine similarity s prethodnim člancima
            current_embedding = embeddings[i:i+1]
            previous_embeddings = embeddings[:i]
            
            similarities = self.compute_similarity_matrix(
                np.vstack([current_embedding, previous_embeddings])
            )[0, 1:]  # Uzmi similarity trenutnog s prethodnim
            
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]
            
            result_df.at[i, 'similarity_score'] = float(max_sim)
            
            if max_sim >= self.threshold_duplicate:
                # Duplikat
                result_df.at[i, 'is_original'] = False
                result_df.at[i, 'duplicate_of'] = int(max_sim_idx)
                result_df.at[i, 'similarity_category'] = 'duplicate'
                
                # Pronađi pravi original (rekurzivno)
                original_idx = self._find_original_index(result_df, max_sim_idx)
                result_df.at[i, 'original_source'] = result_df.at[original_idx, source_col]
                n_duplicates += 1
                
            elif max_sim >= self.threshold_similar:
                # Slično (parafrazirano)
                result_df.at[i, 'is_original'] = False
                result_df.at[i, 'duplicate_of'] = int(max_sim_idx)
                result_df.at[i, 'similarity_category'] = 'similar'
                
                original_idx = self._find_original_index(result_df, max_sim_idx)
                result_df.at[i, 'original_source'] = result_df.at[original_idx, source_col]
                n_similar += 1
        
        logger.info(f"✅ Found {n_duplicates} duplicates, {n_similar} similar articles")
        
        return result_df
    
    def _find_original_index(self, df: pd.DataFrame, idx: int) -> int:
        """Rekurzivno pronađi pravi original."""
        if df.at[idx, 'is_original'] or df.at[idx, 'duplicate_of'] is None:
            return idx
        return self._find_original_index(df, int(df.at[idx, 'duplicate_of']))
    
    def get_spread_analysis(
        self, 
        df: pd.DataFrame,
        source_col: str = 'source'
    ) -> Dict[str, Any]:
        """
        Analiziraj kako se vijesti šire kroz portale.
        
        Args:
            df: DataFrame s rezultatima find_duplicates()
            source_col: Ime kolone s izvorom
            
        Returns:
            Dict s analizom:
            - total_articles: Ukupno članaka
            - original_count: Broj originala
            - duplicate_count: Broj duplikata
            - similar_count: Broj sličnih
            - copy_paste_ratio: Omjer copy-paste novinarstva
            - top_copiers: Portali koji najviše kopiraju
            - top_originators: Portali koji prvi objavljuju
            - spread_chains: Lanci širenja vijesti
        """
        if 'is_original' not in df.columns:
            raise ValueError("DataFrame must have duplicate detection results. "
                           "Run find_duplicates() first.")
        
        total = len(df)
        originals = df[df['similarity_category'] == 'original']
        duplicates = df[df['similarity_category'] == 'duplicate']
        similar = df[df['similarity_category'] == 'similar']
        
        # Statistika po portalima
        copier_counts = Counter(df[~df['is_original']][source_col].tolist())
        originator_counts = Counter(df[df['is_original']]['original_source'].tolist())
        
        # Spread chains - grupiraj po originalnom izvoru
        spread_chains = []
        for orig_idx in df[df['is_original']].index:
            chain_members = df[df['duplicate_of'] == orig_idx]
            if len(chain_members) > 0:
                spread_chains.append({
                    'original': {
                        'title': df.at[orig_idx, 'title'][:100] if 'title' in df.columns else f"Article {orig_idx}",
                        'source': df.at[orig_idx, source_col],
                        'date': str(df.at[orig_idx, 'publishedAt']) if 'publishedAt' in df.columns else 'N/A'
                    },
                    'copies': len(chain_members),
                    'copy_sources': chain_members[source_col].tolist()
                })
        
        # Sortiraj spread chains po broju kopija
        spread_chains = sorted(spread_chains, key=lambda x: x['copies'], reverse=True)
        
        return {
            'total_articles': total,
            'original_count': len(originals),
            'duplicate_count': len(duplicates),
            'similar_count': len(similar),
            'copy_paste_ratio': (len(duplicates) + len(similar)) / total if total > 0 else 0,
            'duplicate_ratio': len(duplicates) / total if total > 0 else 0,
            'similar_ratio': len(similar) / total if total > 0 else 0,
            'top_copiers': copier_counts.most_common(10),
            'top_originators': originator_counts.most_common(10),
            'spread_chains': spread_chains[:20]  # Top 20 chains
        }
    
    def get_similarity_clusters(
        self, 
        df: pd.DataFrame,
        min_cluster_size: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Grupiraj slične članke u klastere.
        
        Args:
            df: DataFrame s rezultatima find_duplicates()
            min_cluster_size: Minimalna veličina klastera
            
        Returns:
            Lista klastera s člancima
        """
        clusters = []
        processed = set()
        
        for idx in df.index:
            if idx in processed:
                continue
            
            # Pronađi sve članke povezane s ovim
            cluster_members = [idx]
            processed.add(idx)
            
            # Dodaj sve koji imaju duplicate_of = idx
            duplicates_of_idx = df[df['duplicate_of'] == idx].index.tolist()
            for dup_idx in duplicates_of_idx:
                if dup_idx not in processed:
                    cluster_members.append(dup_idx)
                    processed.add(dup_idx)
            
            # Ako je ovaj članak duplikat, dodaj i njegov original
            if df.at[idx, 'duplicate_of'] is not None:
                orig_idx = int(df.at[idx, 'duplicate_of'])
                if orig_idx not in processed:
                    cluster_members.append(orig_idx)
                    processed.add(orig_idx)
            
            if len(cluster_members) >= min_cluster_size:
                cluster_df = df.loc[cluster_members].copy()
                clusters.append({
                    'size': len(cluster_members),
                    'articles': cluster_df.to_dict('records'),
                    'sources': cluster_df['source'].unique().tolist() if 'source' in cluster_df.columns else [],
                    'date_range': {
                        'start': str(cluster_df['publishedAt'].min()) if 'publishedAt' in cluster_df.columns else 'N/A',
                        'end': str(cluster_df['publishedAt'].max()) if 'publishedAt' in cluster_df.columns else 'N/A'
                    }
                })
        
        # Sortiraj po veličini
        clusters = sorted(clusters, key=lambda x: x['size'], reverse=True)
        
        return clusters


class DuplicateAnalysis:
    """
    Wrapper klasa za jednostavnu analizu duplikata.
    
    Example:
        >>> analysis = DuplicateAnalysis()
        >>> results = analysis.analyze_file('data/processed/articles.csv')
        >>> print(results['summary'])
    """
    
    def __init__(
        self,
        threshold_duplicate: float = 0.85,
        threshold_similar: float = 0.70
    ):
        """Inicijaliziraj analizu."""
        self.detector = DuplicateDetector(
            threshold_duplicate=threshold_duplicate,
            threshold_similar=threshold_similar
        )
    
    def analyze_file(
        self, 
        file_path: str,
        save_results: bool = True,
        output_suffix: str = '_with_duplicates'
    ) -> Dict[str, Any]:
        """
        Analiziraj CSV datoteku s člancima.
        
        Args:
            file_path: Putanja do CSV datoteke
            save_results: Spremi rezultate u novu datoteku
            output_suffix: Sufiks za izlaznu datoteku
            
        Returns:
            Dict s rezultatima analize
        """
        logger.info(f"Analyzing file: {file_path}")
        
        # Učitaj podatke
        df = pd.read_csv(file_path)
        
        # Pronađi duplikate
        results_df = self.detector.find_duplicates(df)
        
        # Generiraj analizu
        spread_analysis = self.detector.get_spread_analysis(results_df)
        clusters = self.detector.get_similarity_clusters(results_df)
        
        # Spremi rezultate
        if save_results:
            output_path = file_path.replace('.csv', f'{output_suffix}.csv')
            results_df.to_csv(output_path, index=False)
            logger.info(f"✅ Results saved to: {output_path}")
        
        return {
            'dataframe': results_df,
            'summary': spread_analysis,
            'clusters': clusters
        }
    
    def print_report(self, results: Dict[str, Any]) -> None:
        """Ispiši human-readable izvještaj."""
        summary = results['summary']
        
        print("\n" + "=" * 60)
        print("📊 DUPLICATE DETECTION REPORT")
        print("=" * 60)
        
        print(f"\n📰 Total Articles: {summary['total_articles']}")
        print(f"   ✅ Originals: {summary['original_count']} ({100 - summary['copy_paste_ratio']*100:.1f}%)")
        print(f"   📋 Duplicates: {summary['duplicate_count']} ({summary['duplicate_ratio']*100:.1f}%)")
        print(f"   🔄 Similar: {summary['similar_count']} ({summary['similar_ratio']*100:.1f}%)")
        print(f"\n📈 Copy-Paste Ratio: {summary['copy_paste_ratio']*100:.1f}%")
        
        if summary['top_copiers']:
            print("\n🏆 Top Copiers (portals that copy most):")
            for source, count in summary['top_copiers'][:5]:
                print(f"   {source}: {count} copied articles")
        
        if summary['top_originators']:
            print("\n⭐ Top Originators (portals that publish first):")
            for source, count in summary['top_originators'][:5]:
                print(f"   {source}: {count} original articles")
        
        if results.get('clusters'):
            print(f"\n🔗 Found {len(results['clusters'])} story clusters")
            for i, cluster in enumerate(results['clusters'][:3]):
                print(f"   Cluster {i+1}: {cluster['size']} articles from {len(cluster['sources'])} sources")
        
        print("\n" + "=" * 60)


def main():
    """CLI za testiranje duplicate detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect duplicate articles')
    parser.add_argument('file', help='CSV file with articles')
    parser.add_argument('--threshold', type=float, default=0.85, help='Duplicate threshold')
    parser.add_argument('--save', action='store_true', help='Save results')
    
    args = parser.parse_args()
    
    analysis = DuplicateAnalysis(threshold_duplicate=args.threshold)
    results = analysis.analyze_file(args.file, save_results=args.save)
    analysis.print_report(results)


if __name__ == '__main__':
    main()
