import pandas as pd
import sqlite3
import matplotlib
import numpy as np
matplotlib.use('TkAgg')  # Configura o backend para TkAgg
import matplotlib.pyplot as plt
plt.ion()  # Ativa o modo interativo

# Constantes
DATABASE_PATH = 'superstore.db'
DISCOUNT_RATE = 0.85
SALES_THRESHOLD = 1000
TOP_N_SUBCATEGORIES = 12


################################
# Funções de Conexão e Execução de Queries
################################

def connect_to_database():
    """Estabelece conexão com o banco de dados."""
    return sqlite3.connect(DATABASE_PATH)


########
def execute_sql_query(conn, query):
    """Executa uma query SQL e retorna o resultado como DataFrame."""
    return pd.read_sql(query, conn)


################################
# Funções de Plotagem de Gráficos
################################

def plot_bar_chart(df, x_column, y_column, title, rotation=45, color='skyblue'):
    """
    Cria um gráfico de barras com as configurações especificadas.
    
    Args:
        df: DataFrame com os dados
        x_column: Nome da coluna para o eixo x
        y_column: Nome da coluna para o eixo y
        title: Título do gráfico
        rotation: Ângulo de rotação dos labels do eixo x
        color: Cor das barras
    """
    plt.figure(figsize=(20, 10))
    plt.bar(df[x_column], df[y_column], color=color)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()


########
def plot_pie_chart(df, x_column, y_column, title):
    """
    Cria um gráfico de pizza com as configurações especificadas.
    
    Args:
        df: DataFrame com os dados
        x_column: Nome da coluna para os labels
        y_column: Nome da coluna para os valores
        title: Título do gráfico
    """
    plt.figure(figsize=(10, 10))
    plt.pie(df[y_column], labels=df[x_column], autopct='%1.1f%%')
    plt.title(title)
    plt.tight_layout()


########
def plot_line_chart(df, x_column, y_column, title, hue_column=None):
    """
    Cria um gráfico de linha com as configurações especificadas.
    
    Args:
        df: DataFrame com os dados
        x_column: Nome da coluna para o eixo x
        y_column: Nome da coluna para o eixo y
        title: Título do gráfico
        hue_column: Coluna para separar as linhas por cor
    """
    plt.figure(figsize=(20, 10))
    if hue_column:
        for segment in df[hue_column].unique():
            segment_data = df[df[hue_column] == segment]
            plt.plot(segment_data[x_column], segment_data[y_column], label=segment)
        plt.legend()
    else:
        plt.plot(df[x_column], df[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()


########
def plot_grouped_bar_chart(df, category_column, subcategory_column, value_column, title):
    """
    Cria um gráfico de barras agrupadas mostrando a relação entre categorias e subcategorias.
    
    Args:
        df: DataFrame com os dados
        category_column: Nome da coluna de categorias
        subcategory_column: Nome da coluna de subcategorias
        value_column: Nome da coluna de valores
        title: Título do gráfico
    """
    plt.figure(figsize=(15, 8))
    
    # Agrupa os dados por categoria e subcategoria
    grouped_data = df.groupby([category_column, subcategory_column])[value_column].sum().unstack()
    
    # Cria o gráfico de barras agrupadas
    ax = grouped_data.plot(kind='bar', stacked=False)
    
    # Configura o gráfico
    plt.title(title)
    plt.xlabel(category_column)
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.legend(title=subcategory_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()


########
def plot_nested_pie_chart(df, category_column, subcategory_column, value_column, title):
    """
    Cria um gráfico de pizza aninhado, com categorias no anel externo e subcategorias no anel interno.
    
    Args:
        df: DataFrame com os dados
        category_column: Nome da coluna de categorias
        subcategory_column: Nome da coluna de subcategorias
        value_column: Nome da coluna de valores
        title: Título do gráfico
    """
    # Agrupa os dados por categoria e subcategoria
    category_sums = df.groupby(category_column)[value_column].sum()
    subcategory_sums = df.groupby([category_column, subcategory_column])[value_column].sum()
    
    # Cria a figura
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Cores para as categorias (anel externo)
    category_colors = plt.cm.Set3(np.linspace(0, 1, len(category_sums)))
    
    # Cores para as subcategorias (anel interno)
    subcategory_colors = plt.cm.Pastel1(np.linspace(0, 1, len(subcategory_sums)))
    
    # Plota o anel externo (categorias)
    outer_pie = ax.pie(category_sums, 
                      radius=1.3,
                      labels=category_sums.index,
                      autopct='%1.1f%%',
                      pctdistance=0.85,
                      colors=category_colors,
                      wedgeprops=dict(width=0.3))
    
    # Plota o anel interno (subcategorias)
    inner_pie = ax.pie(subcategory_sums, 
                      radius=0.8,
                      labels=subcategory_sums.index.get_level_values(1),
                      autopct='%1.1f%%',
                      pctdistance=0.75,
                      colors=subcategory_colors,
                      wedgeprops=dict(width=0.3))
    
    # Adiciona o título
    plt.title(title, pad=20, fontsize=14)
    
    # Cria uma legenda personalizada
    legend_elements = []
    for i, (category, subcategories) in enumerate(subcategory_sums.groupby(level=0)):
        # Adiciona a categoria
        legend_elements.append(plt.Rectangle((0,0), 1,1, fc=category_colors[i], label=category))
        # Adiciona as subcategorias
        for j, subcat in enumerate(subcategories.index.get_level_values(1)):
            legend_elements.append(plt.Rectangle((0,0), 1,1, fc=subcategory_colors[i*len(subcategories)+j], label=f"  {subcat}"))
    
    # Adiciona a legenda
    plt.legend(handles=legend_elements, 
              title="Categorias e Subcategorias",
              loc="center left",
              bbox_to_anchor=(1, 0.5))
    
    # Ajusta o layout
    plt.tight_layout()


########################
# Funções de Análise
########################

def analyze_office_supplies_sales(conn):
    """Analisa as vendas de produtos da categoria Office Supplies."""
    query = """
        SELECT City, SUM(Sales) as TotalSales
        FROM superstore
        WHERE Category = 'Office Supplies'
        GROUP BY City
        ORDER BY TotalSales DESC
        LIMIT 1
    """
    return execute_sql_query(conn, query)


########
def analyze_sales_by_date(conn):
    """Analisa o total de vendas por data."""
    query = """
        SELECT "Order Date", SUM(Sales) as TotalSales
        FROM superstore
        GROUP BY "Order Date"
        ORDER BY "Order Date"
    """
    return execute_sql_query(conn, query)


########
def analyze_sales_by_state(conn):
    """Analisa o total de vendas por estado."""
    query = """
        SELECT State, SUM(Sales) as TotalSales
        FROM superstore
        GROUP BY State
        ORDER BY TotalSales DESC
    """
    return execute_sql_query(conn, query)


########
def analyze_top_cities(conn, limit=10):
    """Analisa as top cidades com maior total de vendas."""
    query = f"""
        SELECT City, SUM(Sales) as TotalSales
        FROM superstore
        GROUP BY City
        ORDER BY TotalSales DESC
        LIMIT {limit}
    """
    return execute_sql_query(conn, query)


########
def analyze_sales_by_segment(conn):
    """Analisa o total de vendas por segmento."""
    query = """
        SELECT Segment, SUM(Sales) as TotalSales
        FROM superstore
        GROUP BY Segment
        ORDER BY TotalSales DESC
    """
    return execute_sql_query(conn, query)


########
def analyze_sales_by_segment_and_year(conn):
    """Analisa o total de vendas por segmento e por ano."""
    query = """
        SELECT Segment, "Order Date", SUM(Sales) as TotalSales
        FROM superstore
        GROUP BY Segment, "Order Date"
        ORDER BY "Order Date"
    """
    return execute_sql_query(conn, query)


########
def analyze_sales_with_discount_15(conn):
    """Analisa quantas vendas receberiam 15% de desconto (vendas > 1000)."""
    query = f"""
        SELECT COUNT(*) as TotalSales
        FROM superstore
        WHERE Sales > {SALES_THRESHOLD}
    """
    return execute_sql_query(conn, query)


########
def analyze_discount_impact(conn):
    """Analisa o impacto do desconto nas vendas acima do threshold."""
    query = f"""
        SELECT AVG(Sales) as avgBefore, 
               AVG(Sales * {DISCOUNT_RATE}) as avgAfter 
        FROM superstore 
        WHERE Sales > {SALES_THRESHOLD}
    """
    return execute_sql_query(conn, query)


########
def analyze_sales_by_segment_month_year(conn):
    """Analisa a média de vendas por segmento, por ano e por mês."""
    query = """
        SELECT Segment,
               strftime('%Y', "Order Date") as Year,
               strftime('%m', "Order Date") as Month,
               AVG(Sales) as AvgSales
        FROM superstore
        GROUP BY Segment, Year, Month
        ORDER BY Year, Month
    """
    return execute_sql_query(conn, query)


########
def analyze_top_subcategories(conn, limit=TOP_N_SUBCATEGORIES):
    """Analisa as top subcategorias com maior total de vendas."""
    query = f"""
        SELECT Category, "Sub-Category", SUM(Sales) as TotalSales
        FROM superstore
        GROUP BY Category, "Sub-Category"
        ORDER BY TotalSales DESC
        LIMIT {limit}
    """
    return execute_sql_query(conn, query)


########
def main():
    """Função principal que executa todas as análises."""
    conn = connect_to_database()
    
    try:
        # Análise 1: Vendas de Office Supplies. 
        print("\n1. Análise de Vendas de Office Supplies:")
        result = analyze_office_supplies_sales(conn)
        print(result)
        
        # Análise 2: Vendas por Data.
        print("\n2. Análise de Vendas por Data:")
        result = analyze_sales_by_date(conn)
        print(result)
        plot_bar_chart(result, 'Order Date', 'TotalSales', 'Revenue by Order Date', 45, 'skyblue')
        plt.xticks(result['Order Date'][::30], result['Order Date'][::30], rotation=45)  # Uma data a cada 30 dias
        plt.show(block=True)
        
        # Análise 3: Vendas por Estado.
        print("\n3. Análise de Vendas por Estado:")
        result = analyze_sales_by_state(conn)
        print(result)
        plot_bar_chart(result, 'State', 'TotalSales', 'Total Sales by State', 90, 'skyblue')
        plt.show(block=True)
        
        # Análise 4: Top Cidades.
        print("\n4. Análise de Top Cidades:")
        result = analyze_top_cities(conn)
        print(result)
        plot_bar_chart(result, 'City', 'TotalSales', 'Top 10 Cities by Sales', 90, 'skyblue')
        plt.show(block=True)
        
        # Análise 5: Vendas por Segmento. 
        print("\n5. Análise de Vendas por Segmento:")
        result = analyze_sales_by_segment(conn)
        print(result)
        plot_pie_chart(result, 'Segment', 'TotalSales', 'Total Sales by Segment')
        plt.show(block=True)
        
        # Análise 6: Vendas por Segmento e Ano. 
        print("\n6. Análise de Vendas por Segmento e Ano:")
        result = analyze_sales_by_segment_and_year(conn)
        print(result)
        
        # Análise 7: Quantas vendas receberiam 15% de desconto.
        print("\n7. Quantas vendas receberiam 15% de desconto:")
        result = analyze_sales_with_discount_15(conn)
        print(result)

        # Análise 8: Média do valor de venda antes e depois do desconto.
        print("\n8. Média do valor de venda antes e depois do desconto:")
        result = analyze_discount_impact(conn)
        print(result)

        # Análise 9: Média de vendas por segmento, por ano e por mês.
        print("\n9. Média de vendas por segmento, por ano e por mês:")
        result = analyze_sales_by_segment_month_year(conn)
        print(result)
        # Criar uma coluna de data para o gráfico
        result['Date'] = pd.to_datetime(result['Year'] + '-' + result['Month'] + '-01')
        plot_line_chart(result, 'Date', 'AvgSales', 'Average Sales by Segment Over Time', 'Segment')
        plt.show(block=True)
        
        # Análise 10: Total de Vendas Por Categoria e SubCategoria.
        print("\n10. Total de Vendas Por Categoria e SubCategoria, Considerando Somente as Top 12 SubCategorias:")
        result = analyze_top_subcategories(conn)
        print(result)
        plot_nested_pie_chart(result, 'Category', 'Sub-Category', 'TotalSales', 'Distribution of Sales by Category and Subcategory')
        plt.show(block=True)
        
    finally:
        conn.close()


########
if __name__ == "__main__":
    main()

