from collections import Counter

def count_lines(file_path):
    # 读取文件内容
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 去除行尾的换行符
    lines = [line.strip() for line in lines]
    
    # 使用Counter统计每行出现的次数
    line_counts = Counter(lines)
    
    # 按照出现次数从高到低排序
    sorted_counts = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_counts

if __name__ == "__main__":
    file_path = "./a.txt"
    counts = count_lines(file_path)
    
    # 打印结果
    print("每行字符串出现的次数统计:")
    print("="*60)
    for line, count in counts:
        print(f"出现 {count} 次: {line}")
    
    print("\n总计不同行数:", len(counts))
    print(f"总计行数: {sum(count for _, count in counts)}")