using System;
using static System.Math;

class Program
{
    static bool XOR(bool x1, bool x2)
    {
        return (x1 || x2) && !(x1 && x2);
    }

    static void Main()
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Random rnd = new Random();

        Console.WriteLine("Генерація 100 випадкових точок та класифікація за допомогою XOR:");
        Console.WriteLine("--------------------------------------------------------------------");

        // --- Детальний аналіз 1 точки ---
        double x1 = rnd.NextDouble();
        double x2 = rnd.NextDouble();

        bool b1 = x1 >= 0.5;
        bool b2 = x2 >= 0.5;
        bool result = XOR(b1, b2);

        Console.WriteLine($"Точка для аналізу: ({x1:F2}, {x2:F2})");

        Console.WriteLine("\nКрок 1: Перетворення координат у бінарні значення:");
        Console.WriteLine($"- x1 ({x1:F2}) >= 0.5? -> {(b1 ? "Так" : "Ні")}. Бінарне значення: {(b1 ? 1 : 0)}");
        Console.WriteLine($"- x2 ({x2:F2}) >= 0.5? -> {(b2 ? "Так" : "Ні")}. Бінарне значення: {(b2 ? 1 : 0)}");

        Console.WriteLine("\nКрок 2: Застосування логічної функції XOR:");
        Console.WriteLine($"- OR-операція: {(b1 ? 1 : 0)} OR {(b2 ? 1 : 0)} = {(b1 || b2 ? 1 : 0)}");
        Console.WriteLine($"- AND-операція: {(b1 ? 1 : 0)} AND {(b2 ? 1 : 0)} = {(b1 && b2 ? 1 : 0)}");
        Console.WriteLine($"- NOT-операція: NOT ({(b1 && b2 ? 1 : 0)}) = {(!(b1 && b2) ? 1 : 0)}");

        Console.WriteLine("\nКрок 3: Об'єднання результатів:");
        Console.WriteLine($"({(b1 || b2 ? 1 : 0)}) AND ({(!(b1 && b2) ? 1 : 0)}) = {(result ? 1 : 0)}");

        Console.WriteLine($"\nОстаточний результат для ({x1:F2}, {x2:F2}): {result}");
        Console.WriteLine("--------------------------------------------------------------------");

        // --- 99 точок ---
        for (int i = 0; i < 99; i++)
        {
            x1 = rnd.NextDouble();
            x2 = rnd.NextDouble();
            b1 = x1 >= 0.5;
            b2 = x2 >= 0.5;
            result = XOR(b1, b2);
            Console.WriteLine(
                $"Точка ({x1:F2}, {x2:F2}) -> ({(b1 ? 1 : 0)}, {(b2 ? 1 : 0)}) " +
                $"=> результат XOR = {(result ? 1 : 0)}"
            );
        }

        Console.WriteLine("\nКласифікація завершена.");
    }
}