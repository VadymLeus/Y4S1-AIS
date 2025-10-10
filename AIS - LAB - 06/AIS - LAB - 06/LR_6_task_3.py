# Ймовірності для класу "Yes"
p_sunny_yes = 2/9
p_normal_yes = 6/9
p_strong_yes = 3/9

# Ймовірності для класу "No"
p_sunny_no = 3/5
p_normal_no = 1/5
p_strong_no = 3/5

# Загальні ймовірності
p_yes = 9/14
p_no = 5/14

score_yes = p_sunny_yes * p_normal_yes * p_strong_yes * p_yes
score_no = p_sunny_no * p_normal_no * p_strong_no * p_no

print(f"Показник для 'Yes': {score_yes:.5f}")
print(f"Показник для 'No': {score_no:.5f}")

total_score = score_yes + score_no
final_p_yes = score_yes / total_score
final_p_no = score_no / total_score

print("\n--- Результати прогнозу ---")
print(f"Ймовірність, що матч відбудеться ('Yes'): {final_p_yes:.1%}")
print(f"Ймовірність, що матч не відбудеться ('No'): {final_p_no:.1%}")

if final_p_yes > final_p_no:
    print("\nВисновок: Модель прогнозує, що матч відбудеться. ✅")
else:
    print("\nВисновок: Модель прогнозує, що матч НЕ відбудеться. ❌")
