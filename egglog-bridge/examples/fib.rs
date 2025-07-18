use egglog_bridge::{define_rule, ColumnTy, DefaultVal, EGraph, FunctionConfig, MergeFn};
use num_rational::Rational64;

fn main() {
    let mut egraph = EGraph::default();
    let rational_ty = egraph.base_values_mut().register_type::<Rational64>();
    let string_ty = egraph.base_values_mut().register_type::<&'static str>();
    // tables
    let add = egraph.add_table(FunctionConfig {
        schema: vec![ColumnTy::Id, ColumnTy::Id, ColumnTy::Id],
        default: DefaultVal::FreshId,
        merge: MergeFn::UnionId,
        name: "add".into(),
        can_subsume: false,
    });
    // let rule = define_rule! {
    //     [egraph] ((-> (mul x (mul y z)) id)) => ((set (mul (mul x y) z) id))
    // };
}
