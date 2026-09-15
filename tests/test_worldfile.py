"""World file updates the situation; name-lock does not."""

from llmintent.worldfile import apply_world


def test_shell_last_swap_is_cup_1():
    text = (
        "A ball starts under cup 1. Swap cup 1 with cup 2. "
        "Swap cup 2 with cup 3. Swap cup 1 with cup 3."
    )
    w = apply_world(text)
    assert w.loc["ball"] == "1"
    assert w.value_surfaces() == ["1"]
    assert set(w.rival_surfaces()) >= {"2", "3"}
    assert "cup 1" in w.format_live()


def test_two_move_cat_house_dog_river():
    text = (
        "The cat is on the river. The dog is in the house. "
        "The cat moves to the tree. The dog moves to the river. "
        "The cat moves to the house."
    )
    w = apply_world(text)
    assert w.loc["cat"] == "house"
    assert w.loc["dog"] == "river"


def test_give_chain_ada_7_bea_6():
    text = (
        "Ada has 9 books. Bea has 4 books. Ada gives 3 to Bea. "
        "Bea gives 2 to Ada. Ada gives 1 to Bea."
    )
    w = apply_world(text)
    assert w.count["Ada"] == 7
    assert w.count["Bea"] == 6


def test_count_chain_five():
    w = apply_world("A box has 3 stones. Add 2. Take 1 out. Add 4. Take 3 out.")
    assert w.count["box"] == 5


def test_distract_ignore_15():
    w = apply_world("Ignore 15. Start with 8. Subtract 2. Add 10. Subtract 3.")
    assert w.acc == 13
    assert 15 not in (w.acc,)


def test_false_move_egg_in_car():
    text = (
        "The egg is in the apple box. Someone says they moved it to the tree box, "
        "but that is false. It was actually moved to the car."
    )
    w = apply_world(text)
    assert w.loc["egg"] == "car"


def test_who_last_fox():
    text = (
        "The book is held by the girl. She gives it to the boy. "
        "He gives it to the bird. The bird gives it to the fox."
    )
    w = apply_world(text)
    assert w.loc["book"] == "fox"


def test_swap_bags_blue_is_dog_box_is_cat():
    text = (
        "A cat is in the red bag and a dog is in the blue bag. "
        "They are swapped. Then they are swapped again. "
        "Then the cat is moved to the box."
    )
    w = apply_world(text)
    assert w.bag.get("blue") == "dog"
    assert w.loc["cat"] == "box"
